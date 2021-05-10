import os
from configs.test_config import TestConfiguration
from data import create_dataset
from models import create_model
from data import image_folder
import cv2
import numpy as np
import time
import json

def rescale(image, new_min=0, new_max=1):
    image = image.copy().astype(np.float32)
    minn = image.min()
    maxx = image.max()
    image = image - minn

    ratio = (new_max - new_min) / (maxx - minn)
    image *= ratio
    image += new_min

    return image


def save_canvas(canvas, image, idx, img_size=256):
    for i in range(3):
        canvas[:, idx * img_size:(idx + 1) * img_size, i] = rescale(image, 0, 255).astype(np.uint8)

    return canvas


if __name__ == '__main__':
    opt = TestConfiguration().parse()  # get test options

    # make test images
    # image_folder.make_test_dataset(opt.test_data_folder, opt.evaluating_folder)

    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.dataroot = opt.evaluating_folder
    # opt.name = 'mri_pix2pix_gray'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    result_file_name = opt.resut_file_name
    if result_file_name == '':
        from datetime import datetime

        now = datetime.now()
        result_file_name = now.strftime("%m%d%Y%H%M%S")
    result_file_name += '.avi'
    if not os.path.isdir(opt.results_dir):
        os.mkdir(opt.results_dir)
    print('Saving result to ', opt.results_dir + result_file_name)
    video_writer = cv2.VideoWriter(opt.results_dir + result_file_name, cv2.VideoWriter_fourcc(*'MJPG'), 5,
                                   (256 * 3, 256))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        canvas = np.zeros((256, 256 * 3, 3), np.uint8)
        canvas = save_canvas(canvas, visuals['real_A'].cpu().numpy()[0][0], 0)
        canvas = save_canvas(canvas, visuals['fake_B'].cpu().numpy()[0][0], 1)
        canvas = save_canvas(canvas, visuals['real_B'].cpu().numpy()[0][0], 2)
        video_writer.write(canvas)
        if opt.show_feed:
            cv2.imshow('canvas', canvas)
            if cv2.waitKey(1):
                pass
            time.sleep(0.1)
        img = visuals['fake_B'].cpu().numpy()[0][0]

        cv2.imwrite('./results/clm.jpg', rescale(img, 0, 255).astype(np.uint8))

    video_writer.release()