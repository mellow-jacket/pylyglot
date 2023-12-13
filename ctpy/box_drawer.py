import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
from PIL import Image, ImageDraw
import json
import shutil

from .issue import issue

from .ocr_tools import perform_ocr_on_all_images, ocr_on_box

class BoxDrawer:
    def __init__(self, issue = None, start = 0, end = None):
        if issue is None:
            raise ValueError('Please give an issue class to the boxdrawer')
        self.path = issue.path
        
        all_images = sorted(os.listdir(self.path['image']))

        # Ensure start and end are within the bounds of available images
        self.start = max(0, min(start, len(all_images)))
        self.end = min(len(all_images), end) if end is not None else len(all_images)

        # Update self.images to only include images in the specified range
        self.images = all_images[self.start:self.end]
        self.image_names = [x.split('.')[0] for x in self.images]
        self.current_image_index = 0
        self.boxes = []
        self.current_box = []

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
        self.ax.set_xlim([0, self.img.shape[1]])
        self.ax.set_ylim([self.img.shape[0], 0])
        self.fig.canvas.draw_idle()


    def onclick(self,event):

        # Check if the click is outside the button areas
        if event.inaxes != self.button_prev.ax and event.inaxes != self.button_next.ax and event.inaxes != self.button_quit.ax:
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


    def save_and_quit(self,event):
        # Save boxes and OCR results
        self.save_data()
        plt.close()


    def save_data(self):
        # Modify this method to only save the box coordinates
        image_name = self.image_names[self.current_image_index]
        with open(os.path.join(self.path['box_coords'], f'{image_name}.txt'), 'w') as f:
            json.dump(self.boxes, f)

        # # Save image with boxes
        # image = Image.open(os.path.join(self.path['image'], self.images[self.current_image_index]))
        # ocr_image = Image.open(os.path.join(self.path['ocr_image'], self.images[self.current_image_index]))
        # if self.boxes:  # Check if there are boxes to draw
        #     draw = ImageDraw.Draw(image)
        #     draw2 = ImageDraw.Draw(ocr_image)
        #     for box in self.boxes:
        #         draw.rectangle([box[0], box[1]], outline="red", width=5)
        #         draw2.rectangle([box[0], box[1]], outline="red", width=5)
        # image.save(os.path.join(self.path['gpt_image'], self.images[self.current_image_index]))
        # ocr_image.save(os.path.join(self.path['ocr_image'], self.images[self.current_image_index]))


    def quit_handler(self, event):
        # Save boxes and OCR results
        self.save_data()
        # Perform OCR on all images before quitting
        self.perform_ocr_on_all_images()
        plt.close()

    def setup_ui(self):
        # Adjust figure layout to ensure there's space for buttons
        plt.subplots_adjust(bottom=0.2)

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