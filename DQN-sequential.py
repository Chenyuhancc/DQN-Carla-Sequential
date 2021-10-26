
import wandb
from tqdm import tqdm_notebook as tqdm
from threading import Thread
from jupyterplot import ProgressPlot
import tensorflow as tf
from plotnine import *
import pandas as pd
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.models import Sequential
from collections import deque
import datetime
import math
import numpy as np
import time
import cv2 as cv
import random
import sys
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#import keras.backend.tensorflow_backend as backend
wandb.init(name='car', project="carla-rl")

try:
    sys.path.append(glob.glob('/home/chankahou/Downloads/CARLA_0.9.11/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    )
except IndexError:
    print("EGG not found")
pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import Location, Rotation

class CarEnv:
    global town
    actor_list = []
    collision_hist = []
    pt_cloud = []
    pt_cloud_filtered = []

    def __init__(self):
        self.high_res_capture = False
        if self.high_res_capture:
            self.IM_HEIGHT = 240
        else:
            self.IM_HEIGHT = 84
        self.IM_WIDTH = 84*2
        self.recording = True
        try:
            self.client = carla.Client('202.175.25.142', 2000)
            self.client.set_timeout(10.0)
            #world = self.client.load_world(town)
            self.world = self.client.get_world()
            #spectator = self.world.get_spectator()
            #spectator.set_transform(carla.Transform(carla.Location(249, -120, 3), carla.Rotation(yaw=-90)))
            for x in list(self.world.get_actors()):
                if 'vehicle' in x.type_id or 'sensor' in x.type_id:
                    x.destroy()
            blueprint_library = self.world.get_blueprint_library()
            self.Isetta = blueprint_library.filter('model3')[0]
            spawn_point = random.choice(self.world.get_map().get_spawn_points())
            #spawn_point = carla.Transform(Location(x=-115.4, y=4.0, z=1),Rotation(pitch=0, yaw=180, roll=0))
            self.vehicle = self.world.spawn_actor(self.Isetta, spawn_point)
            self.place = 0
            self.start_position = self.get_position()
        except RuntimeError:
            print("Init phase failed, check server connection. Retrying in 30s")
            time.sleep(30)
        #self.collision_hist = []
        self.collision_hist = []
        self.actor_list = []
        self.pt_cloud = []
        self.pt_cloud_filtered = []
        '''
        if self.place == 0:
            transform = carla.Transform(carla.Location(
                249, -130, 0.1), carla.Rotation(0, -90, 0))
        else:
            transform = carla.Transform(carla.Location(
                self.lo_x, self.lo_y), carla.Rotation(0, -90, 0))
        '''
       
        self.actor_list.append(self.vehicle)
        self.lidar_sensor = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_sensor.set_attribute('points_per_second', '100000')
        self.lidar_sensor.set_attribute('channels', '32')
        self.lidar_sensor.set_attribute('range', '10000')
        self.lidar_sensor.set_attribute('upper_fov', '10')
        self.lidar_sensor.set_attribute('lower_fov', '-10')
        self.lidar_sensor.set_attribute('rotation_frequency', '60')
        transform = carla.Transform(carla.Location(x=0, z=1.9))
        time.sleep(0.01)
        self.sensor = self.world.spawn_actor(
            self.lidar_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_lidar(data))

        cam_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))

        cam_bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.s_camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.s_camera)
        self.s_camera.listen(lambda data: self.process_img_semantic(data))

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, brake=0.0))
        self.episode_start = time.time()
        time.sleep(0.4)
        transform2 = carla.Transform(carla.Location(x=2.5, z=0.7))
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(
            colsensor, transform2, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

    def set_location(self, x, y):
        self.lo_x, self.lo_y = x, y
        self.place = x, y

    def get_position(self):
        return self.vehicle.get_location()

    def Black_screen(self):
        settings = self.world.get_settings()
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)

    def get_fps():
        world_snapshot = self.world.get_snapshot()
        fps = 1/world_snapshot.timestamp.delta_seconds
        return fps

    def get_speed(self):    # m/s
        velocity = self.vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def reset(self):
        self.flag = 0
        #spawn_point = random.choice(self.world.get_map().get_spawn_points())
        #self.vehicle = self.world.spawn_actor(self.Isetta, spawn_point)
        #self.transform = carla.Transform(carla.Location(249, -130, 0.1), carla.Rotation(0, -90, 0))
        #self.vehicle = self.world.spawn_actor(self.Isetta, self.transform)
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        #spawn_point = carla.Transform(Location(x=-115.4, y=4.0, z=1),Rotation(pitch=0, yaw=180, roll=0))
        self.vehicle.set_location(spawn_point.location)#self.start_position)
        self.flag = 1
        # while self.distance_to_obstacle_f is None
        while (not hasattr(self,'distance_to_obstacle_f') or
          self.distance_to_obstacle_f is None or
          not hasattr(self,'distance_to_obstacle_r') or
          self.distance_to_obstacle_r is None or
          not hasattr(self,'distance_to_obstacle_l') or
          self.distance_to_obstacle_l is None):
            time.sleep(0.01)
        self.episode_start = time.time()
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, brake=0.0))
        xx = self.distance_to_obstacle_f
        yy = self.distance_to_obstacle_r
        zz = self.distance_to_obstacle_l
        state_ = np.array([xx, yy, zz])
        return state_

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, raw_image):
        self.image = np.array(raw_image.raw_data)
        self.image = self.image.reshape(
            (self.IM_HEIGHT, self.IM_WIDTH, 4))  # RGBA
        self.image = cv.cvtColor(self.image, cv.COLOR_RGBA2RGB)
        if self.high_res_capture and self.recording:
            self.image_array.append(image)
        cv.imshow('image',cv.resize(self.image,(500,500)))
        cv.waitKey(1)
        # trim the top part
        self.image = self.image[int(self.IM_HEIGHT//2.4)::]  
        print("img here")
        return

    def process_img_semantic(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))  # RGBA
        i2 = i2[int(self.IM_HEIGHT//2.4)::] # trim the top part
        self.semantic_image = i2[:, :, 2]
        return

    def process_lidar(self, raw):
        points_new = []
        points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
        for i in range(points.shape[0]//4):
            points_new.append(points[4*i])
            points_new.append(points[4*i+1])
            points_new.append(points[4*i+2])
        points_new = np.asarray(points_new)
        points = np.reshape(
            points_new, (int(points_new.shape[0] / 3), 3))*np.array([1, -1, -1])
        lidar_f = lidar_line(points, 90, 2)
        lidar_r = lidar_line(points, 45, 2)
        lidar_l = lidar_line(points, 135, 2)

        if len(lidar_f) == 0:
            pass
        else:
            self.distance_to_obstacle_f = min(lidar_f[:, 1])-2.247148275375366

        if len(lidar_r) == 0:
            pass
        else:
            self.distance_to_obstacle_r = np.sqrt(
                min(lidar_r[:, 0]**2 + lidar_r[:, 1]**2))

        if len(lidar_l) == 0:
            pass
        else:
            self.distance_to_obstacle_l = np.sqrt(
                min(lidar_l[:, 0]**2 + lidar_l[:, 1]**2))

    def step(self, action):
        global sleepy, steer_
        xx = self.distance_to_obstacle_f
        yy = self.distance_to_obstacle_r
        zz = self.distance_to_obstacle_l
        reward = xx+yy+zz
        reward = reward * 0.004
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, brake=0.0, steer=0))
            time.sleep(sleepy)
            steer_ += 0
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, brake=0.0, steer=0.1))
            time.sleep(sleepy)
            steer_ += 0.1
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, brake=0.0, steer=-0.1))
            time.sleep(sleepy)
            steer_ += -0.1
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, brake=0.0, steer=0.3))
            time.sleep(sleepy)
            steer_ += 0.3
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, brake=0.0, steer=-0.3))
            time.sleep(sleepy)
            steer_ += -0.3

        if len(self.collision_hist) != 0:
            done = True
        else:
            done = False
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        state_ = np.array([xx, yy, zz])
        return state_, reward, done, None


def xxx():
    for x in list(env.world.get_actors()):
        if 'vehicle' in x.type_id or 'sensor' in x.type_id:
            x.destroy()


def lidar_line(points, degree, width):
    angle = degree*(2*np.pi)/360
    points_l = points
    points_l = points_l[np.logical_and(
        points_l[:, 2] > -1.75, points_l[:, 2] < 1000)]  # z
    points_l = points_l[np.logical_and(np.tan(angle)*points_l[:, 0]+width*np.sqrt(1+np.tan(angle)**2) >=
                                       points_l[:, 1], np.tan(angle)*points_l[:, 0]-width*np.sqrt(1+np.tan(angle)**2) <= points_l[:, 1])]  # y
    if 180 > degree > 0:
        points_l = points_l[np.logical_and(
            points_l[:, 1] > 0, points_l[:, 1] < 1000)]  # y>0
    if 180 < degree < 360:
        points_l = points_l[np.logical_and(
            points_l[:, 1] < 0, points_l[:, 1] > -1000)]  # x
    if degree == 0 or degree == 360:
        points_l = points_l[np.logical_and(
            points_l[:, 0] > 0, points_l[:, 0] < 1000)]  # x
    if degree == 180:
        points_l = points_l[np.logical_and(
            points_l[:, 0] > -1000, points_l[:, 0] < 0)]
    return points_l


def save_every_n_episode(num_of_episode):
    n = cum = avg = 0
    avg_loss = []
    for i in Loss[1:]:
        n += 1
        cum += i
        avg = cum/n
        avg_loss.append(avg)

    df = pd.DataFrame({'Episode': ep, 'Reward': ep_rewards,
                       'avg_reward': avg_reward, 'Step': Step, 'Loss': Loss[1:],
                       'avg_loss': avg_loss,
                       'Explore': Explore, 'PCT_Explore': np.array(Explore)/np.array(Step)*100,
                       'Epsilon': Epsilon, 'Dist_stop': Dist_stop,
                       'Max_accel': Max_accel, 'Avg_accel': Avg_accel})
    if LOAD == True:
        df = pd.concat([df_load, df], ignore_index=True)

    name = 'full_deep_brake_cont_reward_limit_action@' + \
        str(num_of_episode)  # INSERT FILE NAME
    n = datetime.datetime.now()
    n = n.strftime('_%m%d%y_%H%M')

    file_path = "data/"
    df.to_csv(file_path+'{}.csv'.format(name))
    agent.save_model(file_path+name)


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
        #self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def get_weight(self):

        w = self.model.get_weights()
        return w

    def predict(self, state):

        predict = self.model.predict(state.reshape((1, self.state_size)))
        return predict

    def save_model(self, name):
        model_json = self.model.to_json()
        with open("{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}.h5".format(name))
        print("Saved model to disk")

    def create_model(self):
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='relu'))
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        global Loss
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            self.terminate = True
            Loss.append(0)
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(
            current_states, PREDICTION_BATCH_SIZE)
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(
            new_current_states, PREDICTION_BATCH_SIZE)
        X = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        history = self.model.fit(np.array(X), np.array(
            y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        Loss.append(history.history['loss'][0])
        self.target_model.set_weights(self.model.get_weights())

    def get_qs(self, state):
        return self.model.predict(state.reshape((1, self.state_size)))[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, self.state_size)).astype(np.float32)
        y = np.random.uniform(size=(1, self.action_size)).astype(np.float32)

        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True
        print('Start Train')
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


class DQNAgent_load_model:

    def __init__(self, state_size, action_size, model):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.loaded_model(model)
        self.target_model = self.loaded_model(model)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
        #self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def loaded_model(self, model):

        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def get_weight(self):

        w = self.model.get_weights()
        return w

    def predict(self, state):

        predict = self.model.predict(state.reshape((1, self.state_size)))
        return predict

    def save_model(self, name):
        model_json = self.model.to_json()
        with open("{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}.h5".format(name))
        print("Saved model to disk")

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        global Loss
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            self.terminate = True
            Loss.append(0)
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(
            current_states, PREDICTION_BATCH_SIZE)
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(
            new_current_states, PREDICTION_BATCH_SIZE)
        X = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        history = self.model.fit(np.array(X), np.array(
            y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        history
        Loss.append(history.history['loss'][0])
        self.target_model.set_weights(self.model.get_weights())

    def get_qs(self, state):
        return self.model.predict(state.reshape((1, self.state_size)))[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, self.state_size)).astype(np.float32)
        y = np.random.uniform(size=(1, self.action_size)).astype(np.float32)

        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True
        print('Start Train')
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


def save_every_5k(name, n):
    df.to_csv('{}_{}.csv'.format(name, n))
    agent.save_model(name+'_'+str(n))


if __name__ == '__main__':

    SECONDS_PER_EPISODE = 100
    REPLAY_MEMORY_SIZE = 5_000
    MIN_REPLAY_MEMORY_SIZE = 32
    MINIBATCH_SIZE = 32
    PREDICTION_BATCH_SIZE = 1
    TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
    UPDATE_TARGET_EVERY = 5
    MEMORY_FRACTION = 0.4
    MIN_REWARD = -200
    EPISODES = 32000
    DISCOUNT = 0.99
    epsilon = 1
    EPSILON_DECAY = 0.99975  # 0.9975 99975
    MIN_EPSILON = 0.01
    AGGREGATE_STATS_EVERY = 10
    state_size = 3
    action_size = 5

    FPS = 60
    town = 'town03'
    ep_rewards = []
    ep = []
    avg = 0
    av_loss = 0
    avg_loss = []
    avg_reward = []
    Step = []
    Loss = []
    Explore = []
    Steer = []
    Epsilon = []
    random.seed(1)
    np.random.seed(1)
    steer_amt = 0.3
    sleepy = 0.3

    agent = DQNAgent(state_size, action_size)
    load_episode = 0

    env = CarEnv()
    agent.train_in_loop()
    agent.get_qs(np.ones((1, state_size)))

    with tqdm(total=EPISODES-load_episode) as pbar:
        for episode in range(EPISODES-load_episode):
            env.collision_hist = []
            episode_reward = 0
            loss = 0
            step = 1
            explore = 0
            steer_ = 0
            current_state = env.reset()
            done = False
            episode_start = time.time()
            while True:
                spectator = env.world.get_spectator()
                spectator_transform = env.vehicle.get_transform()
                spectator_transform.location += carla.Location(x=-2, y=0, z=2.0)
                spectator.set_transform(spectator_transform)
                rand = np.random.random()
                if rand > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                    new_state, reward, done, _ = env.step(action)
                    time.sleep(1/FPS)
                else:
                    action = np.random.randint(0, action_size)
                    new_state, reward, done, _ = env.step(action)
                    explore += 1
                    time.sleep(1/FPS)
                episode_reward += reward
                agent.update_replay_memory(
                    (current_state, action, reward, new_state, done))
                current_state = new_state
                step += 1
                if done:
                    break
            agent.train()

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            print('Episode :{}, Step :{}, Epsilon :{} ,Reward :{}, Explore_rate :{}, loss :{} ,Steer :{}'
                  .format(episode+load_episode, step, epsilon, episode_reward, explore/step, Loss[episode], steer_/step))
            wandb.log({'Reward': episode_reward})
            ep_rewards.append(episode_reward)
            ep.append(episode+load_episode)
            Step.append(step)
            Explore.append(explore)
            Steer.append(steer_/step)
            Epsilon.append(epsilon)
            avg = ((avg*(episode+load_episode)+episode_reward) /
                   (episode+load_episode+1))
            avg_reward.append(avg)
            av_loss = ((av_loss*(episode+load_episode) +
                        Loss[episode])/(episode+load_episode+1))
            wandb.log({'loss': av_loss})
            avg_loss.append(av_loss)
            if (episode+load_episode) % 5000 == 0:
                df = pd.DataFrame({'Episode': ep, 'Reward': ep_rewards,
                                   'avg_reward': avg_reward, 'Step': Step, 'Loss': Loss[1:],
                                   'avg_loss': avg_loss,
                                   'Steer': Steer, 'Explore': Explore, 'PCT_Explore': np.array(Explore)/np.array(Step)*100, 'Epsilon': Epsilon})
                save_every_5k('train_info', episode+load_episode)
    close_carla()
