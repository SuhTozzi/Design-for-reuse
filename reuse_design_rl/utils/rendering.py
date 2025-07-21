
import pygame

import numpy as np
import random, json, glob, os


class render():
    
    def __init__(self, file_name=None):

        self.read_file(file_name)
        self.c_chart = ['mediumorchid','royalblue','turquoise','mediumseagreen','yellowgreen','gold','lightcoral', 'gray']

        # retreive canvas state data
        canvas = np.array(self.data[f"episode_{self.default_ep}"][-2]['canvas state'])
        self.w = canvas.shape[1]
        self.h = canvas.shape[0]
        
        self.scale_factor = min(800//self.w, 600//self.h)
        self.CELL = self.scale_factor
        self.design_size = np.array([self.w, self.h])*self.scale_factor
        self._screen_size = tuple(map(sum, zip(self.design_size, (-1,-1))))
        self.SCREEN_W = self._screen_size[0]
        self.SCREEN_H = self._screen_size[1]

        # pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self._screen_size)
        self.clock = pygame.time.Clock()
        self.font1 = pygame.freetype.Font(r"C:\Windows\Fonts\arial.ttf", 18)
        self.running = True
        pygame.time.wait(50)

    def read_file(self, file_name):
        if file_name == None:
            list_of_files = glob.glob(r'..\result\*.json')
            file_name = max(list_of_files, key=os.path.getmtime)
        else:
            file_name=r'..\result\{}.json'.format(file_name)
        self.file_name = file_name

        with open(self.file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = json.loads(data)
        self.default_ep = int(list(self.data.keys())[-1].split('_')[1])
        self.max_steps = self.data["HYPERPARAM"]["MAX_STEPS"]

        return

    def viz(self, save=False, ep=None, step=None):
        self.ep = self.default_ep if ep==None else ep
        self.step = -1 if step==None else step

        # retreive canvas state data
        self.canvas = np.array(self.data[f"episode_{self.ep}"][self.step-1]['canvas state'])

        time_limit = 1000 
        # Record the start time
        start_time = pygame.time.get_ticks() 
        self.running = True

        while self.running:
            # Quit by time or user input
            elapsed_time = pygame.time.get_ticks() - start_time
            if elapsed_time >= time_limit:
                self.running = False
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            ## Draw the base
            background = pygame.Surface((self.SCREEN_W, self.SCREEN_H)).convert()
            background.fill((0,0,0))  # Black

            self.canvas_layer = pygame.Surface(tuple(self.design_size)).convert_alpha()
            self.canvas_layer.fill((244,243,189,255))  # Yellow-ish, Not transparent (255)
                
            # Draw horizontal lines
            for y in range(self.design_size[1]+1):
                pygame.draw.line(self.canvas_layer, (255,255,255,255), (0,y*self.CELL),
                                (self.SCREEN_W, y*self.CELL))

            # Draw vertical lines
            for x in range(self.design_size[0]+1):
                pygame.draw.line(self.canvas_layer, (255,255,255,255), (x*self.CELL,0),
                                (x*self.CELL, self.SCREEN_H))

            ## Draw the stocks
            inv_keys = np.unique(self.canvas)[1:]
            line_color = (0,0,0,255)
            cell_color = (118,200,147)    # Green-ish, Not transparent (255)
            for inv_key in inv_keys:
                self._draw_stocks(self.canvas_layer, line_color, cell_color, self.canvas, inv_key)

            # Display your work on screen
            self.screen.blit(self.canvas_layer,(0,0))
            pygame.display.update()
            
            self.clock.tick(60)  # limits FPS to 60
            
        if save:
            if step != None:
                if ep != None:
                    pygame.image.save(self.screen, f"{self.file_name[:-5]}_pygame_ep{ep}_step{step}.png")
                else:
                    pygame.image.save(self.screen, f"{self.file_name[:-5]}_pygame_step{step}.png")
            else:
                if ep != None:
                    pygame.image.save(self.screen, f"{self.file_name[:-5]}_pygame_ep{ep}.png")
                else:
                    pygame.image.save(self.screen, f"{self.file_name[:-5]}_pygame.png")
        pygame.quit()

    def viz_ga(self, save=False):
        # retreive canvas state data
        self.canvas = np.array(self.data[f"episode_{self.ep}"][self.step]['canvas state'])

        while self.running:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            ## Draw the base
            background = pygame.Surface((self.SCREEN_W, self.SCREEN_H)).convert()
            background.fill((0,0,0))  # Black

            self.canvas_layer = pygame.Surface(tuple(self.design_size)).convert_alpha()
            self.canvas_layer.fill((244,243,189,255))  # Yellow-ish, Not transparent (255)
                
            # Draw horizontal lines
            for y in range(self.design_size[1]+1):
                pygame.draw.line(self.canvas_layer, (255,255,255,255), (0,y*self.CELL),
                                (self.SCREEN_W, y*self.CELL))

            # Draw vertical lines
            for x in range(self.design_size[0]+1):
                pygame.draw.line(self.canvas_layer, (255,255,255,255), (x*self.CELL,0),
                                (x*self.CELL, self.SCREEN_H))

            ## Draw the stocks
            inv_keys = np.unique(self.canvas)[1:]
            line_color = (0,0,0,255)
            cell_color = (118,200,147)    # Green-ish, Not transparent (255)
            for inv_key in inv_keys:
                self._draw_stocks(self.canvas_layer, line_color, cell_color, self.canvas, inv_key)

            self.screen.blit(self.canvas_layer,(0,0))
            pygame.display.update()
            
            self.clock.tick(60)  # limits FPS to 60
        if save:
            pygame.image.save(self.screen, f"{self.file_name[:-5]}_pygame.png")
        pygame.quit()



    def _draw_inv(self, layer, line_color, cell_color, canvas, inv_key):
        return
    
    def _draw_cur_stock(self, layer, line_color, cell_color, canvas, inv_key):
        return



    def _draw_stocks(self, layer, line_color, cell_color, canvas, inv_key):
        # Find location
        arr = np.argwhere(canvas==inv_key)

        # Color the cells
        for cell in arr:
            self._color_cell(np.flip(cell), color=cell_color)

        # Find the boundary of each stock
        ver_l, ver_r, hor_up, hor_dn = self._find_boundary(arr)

        # Draw horizontal lines: up & down
        self._draw_h_lines(layer, line_color, hor_up)
        self._draw_h_lines(layer, line_color, np.array(hor_dn) + np.array([0,1]))

        # Draw vertical lines: left & right
        self._draw_v_lines(layer, line_color, ver_l)
        self._draw_v_lines(layer, line_color, np.array(ver_r) + np.array([1,0]))

        # Write inventory key
        self._write_text(layer, int(inv_key-100), hor_up[0])

    def _color_cell(self, cell, color=(118,200,147), transparency=20):    
        x = int(cell[0] * self.CELL + 0.5 + 1)
        y = int(cell[1] * self.CELL + 0.5 + 1)
        w = int(self.CELL + 0.5 - 1)
        h = int(self.CELL + 0.5 - 1)
        pygame.draw.rect(self.canvas_layer, color + (transparency,), (x, y, w, h))

    def _find_boundary(self, arr):
        arr_grouped_by_x = [ list(arr[arr[:,0]==i]) for i in np.unique(arr[:,0]) ]
        arr_grouped_by_y = [ list(arr[arr[:,1]==i]) for i in np.unique(arr[:,1]) ]
        
        ver_l = [ np.flip(arr_x[0]) for arr_x in arr_grouped_by_x ]
        ver_r = [ np.flip(arr_x[-1]) for arr_x in arr_grouped_by_x ]
        hor_up = [ np.flip(arr_x[0]) for arr_x in arr_grouped_by_y ]
        hor_dn = [ np.flip(arr_x[-1]) for arr_x in arr_grouped_by_y ]

        return ver_l, ver_r, hor_up, hor_dn

    def _draw_h_lines(self, canvas_layer, line_color, arr):
        for xy in arr:
            pygame.draw.line(canvas_layer, line_color, (xy[0]*self.CELL, xy[1]*self.CELL),
                            ((xy[0]+1)*self.CELL, xy[1]*self.CELL))
            
    def _draw_v_lines(self, canvas_layer, line_color, arr):
        for xy in arr:
            pygame.draw.line(canvas_layer, line_color, (xy[0]*self.CELL, xy[1]*self.CELL),
                            (xy[0]*self.CELL, (xy[1]+1)*self.CELL))
        
    def _write_text(self, canvas_layer, inv_key, text_xy):
        text = f'{inv_key}'
        textRect = self.font1.get_rect(text, size=24)
        textRect.center = ((text_xy[0]+0.5)*self.CELL, (text_xy[1]+0.5)*self.CELL)
        self.font1.render_to(canvas_layer, textRect, text, (0,0,0,255))

