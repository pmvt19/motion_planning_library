import pygame 

class XboxController:

    class XboxControls():
        LTHUMBX = 0
        LTHUMBY = 1
        RTHUMBX = 2
        RTHUMBY = 3
        LTRIGGER = 4
        RTRIGGER = 5
        A = 6
        B = 7
        X = 8
        Y = 9
        LBUMPER = 10
        RBUMPER = 11
        LJOY = 12
        RJOY = 13
        BACK = 14
        START = 15
        XBOX = 16
        DPAD = 17

    class PyGameAxis():
        LTHUMBX = 0
        LTHUMBY = 1
        RTHUMBX = 2
        RTHUMBY = 3
        LTRIGGER = 4
        RTRIGGER = 5

    class PyGameButtons():
        A = 0
        B = 1
        X = 2
        Y = 3
        LBUMPER = 4
        RBUMPER = 5
        BACK = 6
        START = 7
        LJOY = 8
        RJOY = 9
        XBOX = 10

    AXISCONTROLMAP = {PyGameAxis.LTHUMBX: XboxControls.LTHUMBX,
                    PyGameAxis.LTHUMBY: XboxControls.LTHUMBY,
                    PyGameAxis.RTHUMBX: XboxControls.RTHUMBX,
                    PyGameAxis.RTHUMBY: XboxControls.RTHUMBY, 
                    PyGameAxis.RTRIGGER: XboxControls.RTRIGGER,
                    PyGameAxis.LTRIGGER: XboxControls.LTRIGGER}

    BUTTONCONTROLMAP = {PyGameButtons.A: XboxControls.A,
                        PyGameButtons.B: XboxControls.B,
                        PyGameButtons.X: XboxControls.X,
                        PyGameButtons.Y: XboxControls.Y,
                        PyGameButtons.LBUMPER: XboxControls.LBUMPER,
                        PyGameButtons.RBUMPER: XboxControls.RBUMPER,
                        PyGameButtons.BACK: XboxControls.BACK,
                        PyGameButtons.START: XboxControls.START,
                        PyGameButtons.XBOX: XboxControls.XBOX,
                        PyGameButtons.LJOY: XboxControls.LJOY,
                        PyGameButtons.RJOY: XboxControls.RJOY}
    def __init__(self, my_game):

        self.game = my_game
        self.allow_continuous_update = True 

        # LTHUMBX, LTHUMBY, RTHUMBX, RTHUMBY, LTRIGGER, RTRIGGER, A, B, X, Y, LBUMPER, RBUMPER, LJOY, RJOY, BACK, START, XBOX, DPAD
        self.state = [0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0, 0)]

    def get_contoller_state(self):
        return self.state 

    def update_state(self):
        # print("START")
        for event in self.game.event.get():
            if event.type == self.game.JOYBUTTONDOWN:
                pressed_button = event.button 
                state_ind = self.BUTTONCONTROLMAP[pressed_button]
                self.state[state_ind] = 1 
            
            if event.type == self.game.JOYBUTTONUP:
                pressed_button = event.button 
                state_ind = self.BUTTONCONTROLMAP[pressed_button]
                self.state[state_ind] = 0

            if event.type == self.game.JOYHATMOTION:
                self.state[self.XboxControls.DPAD] = event.value

            if event.type == self.game.JOYAXISMOTION:
                # print(event)
                moved_axis = event.axis
                state_ind = self.AXISCONTROLMAP[moved_axis]
                # print(moved_axis, state_ind)
                self.state[state_ind] = event.value 
        return 

    def continuous_update_state(self):
        # clock = self.game.time.Clock()

        while (self.allow_continuous_update):
            # clock.tick(60)
            self.update_state()

    def disable_continuous_update(self):
        self.allow_continuous_update = False 

    def enable_continuous_update(self):
        self.allow_continuous_update = True 