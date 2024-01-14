import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
from PIL import Image, ImageDraw
import json
import shutil

from .issue import issue

from .ocr_tools import perform_ocr_on_all_images, ocr_on_box
from .splits import natural_sort_key

class BoxDrawer:
    def __init__(self, issue = None, start = 0, end = None):
        if issue is None:
            raise ValueError('Please give an issue class to the boxdrawer')
        self.path = issue.path
        
        all_images = sorted(os.listdir(self.path['image']), key=natural_sort_key)

        # Ensure start and end are within the bounds of available images
        self.start = max(0, min(start, len(all_images)))
        self.end = min(len(all_images), end) if end is not None else len(all_images)

        # Update self.images to only include images in the specified range
        self.images = all_images[self.start:self.end]
        
        self.image_names = [x.split('.')[0] for x in self.images]
        self.current_image_index = 0
        self.boxes = []
        self.current_box = []
        self.is_drawing_enabled = False  # State variable for box drawing mode

    def toggle_drawing_mode(self, event):
        """Toggle the box drawing mode on and off."""
        self.is_drawing_enabled = not self.is_drawing_enabled
        print("Drawing mode:", "Enabled" if self.is_drawing_enabled else "Disabled")

    def load_boxes(self):
        """ Load existing box coordinates for the current image from a file. """
        image_name = self.image_names[self.current_image_index]
        box_file_path = os.path.join(self.path['box_coords'], f'{image_name}.txt')
        if os.path.exists(box_file_path):
            with open(box_file_path, 'r') as f:
                self.boxes = json.load(f)
        else:
            self.boxes = []

    def render_boxes(self):
        """ Render the loaded boxes on the current image. """
        for box in self.boxes:
            rect = patches.Rectangle((box[0][0], box[0][1]), box[1][0] - box[0][0], box[1][1] - box[0][1], linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)

    def clear_boxes(self, event):
        """ Clear all boxes from the current image. """
        self.boxes = []
        self.update_image()

    def perform_ocr_on_all_images(self):
        perform_ocr_on_all_images(self)

    def ocr_on_box(self,image, box, lang='kor'):
        ocr_on_box(self,image, box, lang=lang)

    def update_image(self):
        # Clear the axes completely
        self.ax.clear()

        # Load and display the new image
        self.img = plt.imread(os.path.join(self.path['image'], self.images[self.current_image_index]))
        self.image_display = self.ax.imshow(self.img)

        # Reset the plot limits and redraw the canvas
        #self.ax.set_xlim([0, self.img.shape[1]])
        #self.ax.set_ylim([self.img.shape[0], 0])
        self.fig.canvas.draw_idle()
        self.render_boxes()
        # rect = patches.Rectangle((50, 50), 100 - 50, 100 - 50, linewidth=1, edgecolor='r', facecolor='none')
        # self.ax.add_patch(rect)

    def onclick(self,event):

        # Check if the click is outside the button areas
        if self.is_drawing_enabled and event.inaxes == self.ax:
            # Check if the click is outside the button areas
            if event.xdata is not None and event.ydata is not None:
                self.current_box.append((event.xdata, event.ydata))

                # If two points are selected, draw the box
                if len(self.current_box) == 2:
                        # Sort the coordinates to get top-left and bottom-right
                        start_x, end_x = sorted([self.current_box[0][0], self.current_box[1][0]])
                        start_y, end_y = sorted([self.current_box[0][1], self.current_box[1][1]])

                        # Create and add the rectangle
                        box = patches.Rectangle((start_x, start_y), end_x - start_x, end_y - start_y, linewidth=1, edgecolor='r', facecolor='none')
                        self.ax.add_patch(box)

                        # Save the box coordinates as top-left and bottom-right
                        self.boxes.append([(start_x, start_y), (end_x, end_y)])
                        self.current_box = []
                        plt.draw()

    def navigate_images(self, step):
        # Save data before moving to the next or previous image
        self.save_data()

        # Update the current image index
        self.current_image_index += step

        # Check if the current image is the last one
        if self.current_image_index >= len(self.images):
            self.current_image_index-=1
            self.quit_handler(None)  # Save and quit if at the end
        else:
            # Ensure the index stays within bounds
            self.current_image_index = max(0, min(self.current_image_index, len(self.images) - 1))

            # Clear boxes for the new image
            self.boxes = []
            self.update_image()
        self.load_and_render_boxes()

    def save_and_quit(self,event):
        # Save boxes and OCR results
        self.save_data()
        plt.close()


    def save_data(self):
        # Modify this method to only save the box coordinates
        image_name = self.image_names[self.current_image_index]
        with open(os.path.join(self.path['box_coords'], f'{image_name}.txt'), 'w') as f:
            json.dump(self.boxes, f)


    def quit_handler(self, event):
        # Save boxes and OCR results
        self.save_data()
        # Perform OCR on all images before quitting
        self.perform_ocr_on_all_images()
        plt.close()

    def setup_ui(self):
        # Adjust figure layout to ensure there's space for buttons
        plt.subplots_adjust(bottom=0.3)

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Adjust the position of the buttons
        button_height = 0.1  # Increase if more space is needed
        self.prev_button_ax = plt.axes([0.59, 0.1, 0.1, button_height])
        self.next_button_ax = plt.axes([0.7, 0.1, 0.1, button_height])
        self.quit_button_ax = plt.axes([0.81, 0.1, 0.1, button_height])

        self.button_prev = plt.Button(self.prev_button_ax, 'Previous')
        self.button_next = plt.Button(self.next_button_ax, 'Next')
        self.button_quit = plt.Button(self.quit_button_ax, 'Quit')

        self.button_prev.on_clicked(lambda event: self.navigate_images(-1))
        self.button_next.on_clicked(lambda event: self.navigate_images(1))
        self.button_quit.on_clicked(self.quit_handler)

        # Adjust the position of the new buttons
        load_button_ax = plt.axes([0.48, 0.1, 0.1, button_height])
        clear_button_ax = plt.axes([0.37, 0.1, 0.1, button_height])

        self.button_load = plt.Button(load_button_ax, 'Load')
        self.button_clear = plt.Button(clear_button_ax, 'Clear')

        self.button_load.on_clicked(lambda event: self.load_and_render_boxes())
        self.button_clear.on_clicked(self.clear_boxes)

        # Add a button for toggling the drawing mode
        toggle_button_ax = plt.axes([0.26, 0.1, 0.1, 0.1])  # Adjust position and size as needed
        self.button_toggle_drawing = plt.Button(toggle_button_ax, 'Toggle Drawing')
        self.button_toggle_drawing.on_clicked(self.toggle_drawing_mode)



    def load_and_render_boxes(self):
        """ Load and render boxes for the current image. """
        self.load_boxes()
        self.render_boxes()
        plt.draw()


    def draw(self):
        self.fig, self.ax = plt.subplots()
        self.img = plt.imread(os.path.join(self.path['image'], self.images[self.current_image_index]))
        self.image_display = self.ax.imshow(self.img)
        
        self.setup_ui()

        plt.show()
    
    def load(self, name='name'):
        # Define the source directory based on the provided name
        src_directory = os.path.join('box_saves', name)

        # Destination directory from the class's path attribute
        dest_directory = self.path['box_coords']
        # Copy each file from the source directory to the destination directory
        for filename in os.listdir(src_directory):
            file_path = os.path.join(src_directory, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, dest_directory)

    def save(self, name='name'):
        # Define the source directory based on the provided name
        dest_directory = os.path.join('box_saves', name)
        if not os.path.exists(dest_directory):
            os.mkdir(dest_directory)
        # Destination directory from the class's path attribute
        src_directory = self.path['box_coords']
        # Copy each file from the source directory to the destination directory
        for filename in os.listdir(src_directory):
            file_path = os.path.join(src_directory, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, dest_directory)