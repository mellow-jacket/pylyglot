import os
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


def get_files(directory):
    # Get all JPEG files and sort them by page number
    files = [f for f in os.listdir(directory) if f.endswith('.jpeg')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files

def split_and_combine(split_path, split_pos):
    # Split the specified image and combine with the next image
    if os.path.sep in split_path:
        directory = os.path.sep.join(split_path.split(os.path.sep)[:-1])
    else:
        directory = os.path.sep.join(split_path.split('/')[:-1])
    split_file = split_path.split(os.path.sep)[-1]
    file_index = split_file.split('.jp')[0]
    file_index = file_index.split('_num_')[-1]
    file_index = int(file_index)

    combined_bottom = os.path.join(directory,f'page_num_{file_index+1}.jpeg')
    m_ind = 1
    while not os.path.exists(os.path.join(directory,f'page_num_{file_index-m_ind}.jpeg')):
        print('looking for : ', os.path.join(directory,f'page_num_{file_index-m_ind}.jpeg'))
        m_ind+=1
    combined_top = os.path.join(directory,f'page_num_{file_index-m_ind}.jpeg')

    # Split the specified image
    img1 = Image.open(split_path)
    top = img1.crop((0, 0, img1.width, split_pos))
    bottom = img1.crop((0, split_pos, img1.width, img1.height))
    img2 = Image.open(combined_bottom)
    img0 = Image.open(combined_top)

    # Combine with the bottom image
    next_top = img2
    combined2 = Image.new('RGB', (img2.width, bottom.height + next_top.height))
    combined2.paste(bottom, (0, 0))
    combined2.paste(next_top, (0, bottom.height))
    combined2.save(combined_bottom)

    # Correctly combine with the top image
    prev_bottom = img0#.crop((0, img0.height - top.height, img0.width, img0.height))
    combined0 = Image.new('RGB', (img1.width, top.height + prev_bottom.height))
    combined0.paste(prev_bottom, (0, 0))
    combined0.paste(top, (0, prev_bottom.height))
    combined0.save(combined_top)

    # Remove the original split image
    os.remove(split_path)

def on_click(event,path):
    global current_ind
    current_ind+=1
    # Event handler for mouse click
    print('path is : ', path)
    success = split_and_combine(path, event.y)
    if success:
        print("Image split and combined successfully.")
    root.destroy()

# Modify the on_click function to use separate_file
def on_click2(event, path):
    global current_ind
    current_ind += 1
    _separate_file(path, event.y)
    print("Image separated into two and files renamed successfully.")
    root.destroy()

def main(path, on_next = None):
    # Main function to open the specified image and wait for a click or press "Next" button
    directory = os.path.sep.join(path.split('/')[:-1])
    filename = path.split(os.path.sep)[-1]
    global root
    if os.path.exists(path):
        root = tk.Tk()

        # Create and pack the "Next" button at the top
        next_button = tk.Button(root, text="Next", command=on_next)
        next_button.pack(side="top")

        # Load and pack the image
        img = Image.open(path)
        tk_img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=tk_img)
        panel.pack(side="top", fill="both", expand="yes")
        panel.bind("<Button-1>", lambda event: on_click(event, path))

        root.mainloop()
    else:
        print(f"File {filename} not found in the directory.")

def fill_number_gaps(directory):
    # Get and sort the filenames
    filenames = os.listdir(directory)
    sorted_filenames = sorted(filenames, key=natural_sort_key)

    # Ensure the numbering starts from the lowest number
    expected_num = int(re.search(r'\d+', sorted_filenames[0]).group())

    for filename in sorted_filenames:
        #print('checking : ', filename)
        # Extract the current number from the filename
        current_num = int(re.search(r'\d+', filename).group())

        # Check if the current number matches the expected number
        if current_num != expected_num:
            # Construct the new filename with the expected number
            new_filename = filename.replace(str(current_num), str(expected_num))
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        # Increment the expected number for the next iteration
        expected_num += 1

def split_file(path):
    # Main function to open the specified image and wait for a click or press "Next" button
    directory = os.path.sep.join(path.split('/')[:-1])
    filename = path.split(os.path.sep)[-1]
    global root
    global current_ind
    current_ind=0
    if os.path.exists(path):
        root = tk.Tk()

        # Load and pack the image
        img = Image.open(path)
        tk_img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=tk_img)
        panel.pack(side="top", fill="both", expand="yes")
        panel.bind("<Button-1>", lambda event: on_click(event, path))

        root.mainloop()
        #print('checking directory : ', directory)
        fill_number_gaps(directory)
    else:
        print(f"File {filename} not found in the directory.")

def _separate_file(path, event_y):
    directory = os.path.sep.join(path.split('/')[:-1])
    filename = path.split('/')[-1]
    base_name = '_'.join(filename.split('_')[:-1])
    file_number = int(filename.split('_')[-1].split('.')[0])

    # Load and split the image
    img = Image.open(path)
    top = img.crop((0, 0, img.width, event_y))
    bottom = img.crop((0, event_y, img.width, img.height))

    # Temporarily save the bottom half
    temp_bottom_filename = "temp_bottom_image.jpeg"
    bottom.save(os.path.join(directory, temp_bottom_filename))

    # Save the top half with the original file name
    top.save(os.path.join(directory, filename))

    # Increment file names for subsequent images
    increment_file_names(directory, file_number)

    # Rename the bottom half to its final name
    final_bottom_filename = f"{base_name}_{file_number + 1}.jpeg"
    os.rename(os.path.join(directory, temp_bottom_filename),
              os.path.join(directory, final_bottom_filename))
    
def separate_file(path):
    directory = os.path.sep.join(path.split('/')[:-1])
    filename = path.split(os.path.sep)[-1]
    global root
    global current_ind
    current_ind = 0
    if os.path.exists(path):
        root = tk.Tk()

        img = Image.open(path)
        tk_img = ImageTk.PhotoImage(img)
        panel = Label(root, image=tk_img)
        panel.pack(side="top", fill="both", expand="yes")
        panel.bind("<Button-1>", lambda event: on_click2(event, path))

        root.mainloop()
    else:
        print(f"File {filename} not found in the directory.")

def increment_file_names(directory, starting_number):
    files = [f for f in os.listdir(directory) if f.startswith('page_num_') and f.endswith('.jpeg')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for file in reversed(files):
        number = int(file.split('_')[-1].split('.')[0])
        if number > starting_number:
            new_number = number + 1
            new_filename = file.replace(f"_{number}.jpeg", f"_{new_number}.jpeg")
            os.rename(os.path.join(directory, file), os.path.join(directory, new_filename))
