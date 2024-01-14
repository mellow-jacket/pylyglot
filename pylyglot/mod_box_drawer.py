import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
from PIL import Image
import json
import shutil
import ipywidgets as widgets
from IPython.display import display, clear_output

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
        self.fig, self.ax = plt.subplots(figsize=(4,8))
        self.img = plt.imread(os.path.join(self.path['image'], self.images[self.current_image_index]))
        self.image_display = self.ax.imshow(self.img)
    
            # Optimize the plot layout
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Create ipywidgets buttons
        prev_button = widgets.Button(description='Previous')
        next_button = widgets.Button(description='Next')
        quit_button = widgets.Button(description='Quit')

        prev_button.on_click(lambda event: self.navigate_images(-1))
        next_button.on_click(lambda event: self.navigate_images(1))
        quit_button.on_click(lambda event: self.quit_handler(event))

        # Display the buttons in a horizontal layout
        buttons = widgets.HBox([prev_button, next_button, quit_button])
        display(buttons)

        plt.show()


    def draw(self):
        self.setup_ui()
    
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

    def save(self, name = 'testboxsave'):
        self.src_directory = self.path['box_coords']
        self.dest_directory = os.path.join('box_saves', name)

        if os.path.exists(self.dest_directory):
            # Create and display a confirmation button
            confirm_button = widgets.Button(description='Confirm Save', button_style='success')
            cancel_button = widgets.Button(description='Whoops, stop overwrite', button_style='warning')
            confirm_button.on_click(self.confirm_save)
            cancel_button.on_click(self.cancel_save)
            button_box = widgets.HBox([confirm_button, cancel_button])
            display(button_box)
        else:
            self.execute_save()

    def confirm_save(self, button):
        clear_output()
        self.execute_save()
 
    def cancel_save(self, button):
        clear_output()
        print('Aborting save')


    def execute_save(self):
        # Ensure the destination directory exists
        if not os.path.exists(self.dest_directory):
            os.makedirs(self.dest_directory)
        else:
            # If it exists, clear its contents
            for filename in os.listdir(self.dest_directory):
                file_path = os.path.join(self.dest_directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        # Copy each file from the source directory to the destination directory
        for filename in os.listdir(self.src_directory):
            file_path = os.path.join(self.src_directory, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, self.dest_directory)

        print('Save operation completed.')