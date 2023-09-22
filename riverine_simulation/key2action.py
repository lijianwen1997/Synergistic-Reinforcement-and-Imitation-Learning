from pynput import keyboard
import random


class Key2Action:
    def __init__(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.last_key = None

    def on_press(self, key):
        try:
            pass
            # print('alphanumeric key {0} pressed'.format(key.char))
        except AttributeError:
            print('special key {0} pressed'.format(key))

    def on_release(self, key):
        # print('{0} released'.format(key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        self.last_key = key.char

    def get_multi_discrete_action(self):
        action = [1] * 4
        if self.last_key is None:
            return action

        if self.last_key == 'w':
            action[0] = 0
        elif self.last_key == 's':
            action[0] = 2
        elif self.last_key == 'a':
            action[1] = 0
        elif self.last_key == 'd':
            action[1] = 2
        elif self.last_key == 'i':
            action[2] = 0
        elif self.last_key == 'k':
            action[2] = 2
        elif self.last_key == 'j':
            action[3] = 0
        elif self.last_key == 'l':
            action[3] = 2
        else:
            print(f'Unrecognized key {self.last_key}')

        self.last_key = None
        return action

    def get_discrete_action(self):
        action = 0
        if self.last_key is None:
            return [action]

        if self.last_key == 'w':
            action = 1
        elif self.last_key == 's':
            action = 2
        elif self.last_key == 'a':
            action = 3
        elif self.last_key == 'd':
            action = 4
        elif self.last_key == 'i':
            action = 5
        elif self.last_key == 'k':
            action = 6
        elif self.last_key == 'j':
            action = 7
        elif self.last_key == 'l':
            action = 8
        else:
            print(f'Unrecognized key {self.last_key}')

        self.last_key = None
        return [action]

    def get_random_action(self, multi_discrete: bool = False):
        if multi_discrete:  # one-hot
            action = [1] * 4
            axis = random.randint(0, 3)
            direction = random.randint(0, 2)
            action[axis] = direction
            return action
        else:
            return [random.randint(0, 8)]


if __name__ == '__main__':
    # ...or, in a non-blocking fashion:
    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()

    # Collect events until released
    # with keyboard.Listener(
    #         on_press=on_press,
    #         on_release=on_release) as listener:
    #     listener.join()

    k2a = Key2Action()


