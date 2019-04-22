import gym
import math
def sigmoid(x):
     return 1 / (1 + math.e ** -x)

class EnvStreetFighterII(gym.Wrapper):

    def __init__(self, env, agent_name="default", show_env=False):
        super(EnvStreetFighterII, self).__init__(env)

        self.show_env = show_env
        self._obs = None
        self._rew = None
        self._info = {'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,
                      'health': 176}
        self.counted = False
        self.lastHundred = []
        self.win_nb = 0
        self.lose_nb = 0
        self.timeout_nb = 0
        self.env.reward_range = (-5, 10)
        #tensorboard writer
        #self.writer = SummaryWriter(comment="-"+HYPERPARAMS['run_name']+"-"+agent_name)

    def reset(self):
        self.lastHundred = []
        self.win_nb = 0
        self.lose_nb = 0
        self.timeout_nb = 0
        self.episode_reward = 0

        self.counted = False
        self.done = False

        self._info = {'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,
                      'health': 176}

        return self.env.reset()

    def step(self, action):
        #print(self.env.get_action_meaning(action))
        # get reward
        self._obs, _, self.done, _info = self.env.step(action)
        #self._rew = (((self._info['health'] - _info['health'])-(self._info['enemy_health'] - _info['enemy_health'])) / 176)
        delta = (((self._info['enemy_health'] - _info['enemy_health']) - (self._info['health'] - _info['health'])))

        self._rew = (sigmoid(delta / 20)) - 0.5

        self._info = _info
        #Things to do when episode ends
        # if both players got more than 0 energy you still playing
        if self._info['enemy_health'] > 0 and self._info['health'] > 0:
            self.counted = False
        # check resilient energy and update wins
        if not self.counted:
            # Player Wins!
            if self._info['enemy_health'] < 0:
                self.counted = True
                self.win_nb += 1
                self._rew = 1
                self.lastHundred.append(1)
            # Enemy Wins!
            elif self._info['health'] < 0:
                self.counted = True
                self.lose_nb += 1
                #self._rew = -1
                self.lastHundred.append(-1)
            # Timeout
            if self.done:
                self.counted = True
                self.timeout_nb += 1
            # clean up last hundreds
            if len(self.lastHundred) > 100:
                self.lastHundred.pop(0)
        self.episode_reward += self._rew
        #if self._rew != 0.0:
        #    print("delta: ", delta, "Sigmoid: ",self._rew)
        return self._obs, self._rew, self.done, self._info

    def get_episode_reward(self):
        return round(self.episode_reward * self._info['score'], 6)

class SFIIActionWrapper(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SFIIActionWrapper, self).__init__(env)

        # Define Street Fighter II actions
        neutral = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        right = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        left = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        crouch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        crouch_right = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        crouch_left = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        jump_neutral = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        jump_right = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        jump_left = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

        standing_low_punch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        standing_medium_punch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        standing_high_punch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        standing_low_kick = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        standing_medium_kick = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        crouching_low_punch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        crouching_medium_punch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        crouching_high_punch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        crouching_low_kick = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        crouching_medium_kick = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self._actions = [neutral, right, left, crouch, crouch_right, crouch_left, jump_neutral, jump_right, jump_left,
                standing_low_punch, standing_medium_punch, standing_high_punch, standing_low_kick,
                standing_medium_kick,
                crouching_low_punch, crouching_medium_punch, crouching_high_punch, crouching_low_kick,
                crouching_medium_kick]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()