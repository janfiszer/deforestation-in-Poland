import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import visualization


def segment(sample_image, model, model_input_size, white_balance=0, display_images=False):
    """
    Function allowing to apply a segmentation model on an image which shape is bigger that the input.
    It is done so by patching the input into model_input_size and predicting on them.
    Then it is merged and returned. Note: in the input image is not divisible by model_input_size, the result image is
    just truncated (possible improvement in the future)
    :param sample_image: The big image to segment.
    :param model: Segmentation model.
    :param model_input_size: A tuple representing the input size of given model
    :param white_balance: a int from 0 to 100 that describes the strength of white balancing.
    0 the strongest, 100 the weakest.
    :param display_images: A flag whether to display images that are predicted at the moment.
    :return: A full mask of segmented images. Might be truncated
    """
    input_stride = model_input_size[0]

    if white_balance > 0:
        sample_image = visualization.white_balancing(sample_image, percentile_value=white_balance)

    full_mask_size_x = int(sample_image.shape[0] * model_input_size[0]/input_stride)
    full_mask_size_y = int(sample_image.shape[1] * model_input_size[0]/input_stride)
    full_mask = np.zeros((full_mask_size_x, full_mask_size_y))

    # # number of segmentation s to perform
    # n_cuts_x = int(full_mask_size_x * model_input_size[0])
    # n_cuts_y = int(full_mask_size_y * model_input_size[0])

    print(full_mask_size_x, full_mask_size_y)

    for x0 in range(input_stride, full_mask_size_x, input_stride):
        for y0 in range(input_stride, full_mask_size_y, input_stride):
            print(x0, y0)

            patch = sample_image[x0 - input_stride:x0, y0 - input_stride:y0, :]

            # preparing the patch as the U-Net input
            image = tf.keras.utils.array_to_img(patch)
            batch = np.array([tf.image.resize(image, size=model_input_size)])

            if display_images:
                plt.imshow(image)

            # prediction
            pred_mask = model.predict(batch, verbose=0)

            # mapping the results in the correct place
            full_mask[x0 - input_stride:x0, y0 - input_stride:y0] = pred_mask.reshape(model_input_size)
    
    return full_mask

# TODO: make it work for stride != model_input_size
# def classify(sample_image, model, model_input_size, input_stride, white_balance=False, display_images=False):

#     # todo followup
#     input_stride = model_input_size[0]

#     n_cuts = int(sample_image.shape[0]/input_stride)
    
#     result_size = int(sample_image.shape[0] * model_input_size[0]/input_stride)
#     result = np.zeros((result_size, result_size))

#     for i in range(n_cuts):
#         for j in range(n_cuts):
#             patch = sample_image[i*input_stride:(i+1)*input_stride, j* input_stride:(j+1) * input_stride, :]
#             # print(patch.shape)
#             if white_balance:
#                 patch = visualization.white_balancing(patch)

#             image = tf.keras.utils.array_to_img(patch)
#             batch = np.array([tf.image.resize(image, size=model_input_size)])


#             if display_images:
#                 plt.imshow(image)

#             y_pred = model.predict(batch, verbose=0)
#             predicted_class = np.argmax(y_pred)

#             patch_result = np.ones(model_input_size) * predicted_class
#             result[i*input_stride:(i+1)*input_stride, j* input_stride:(j+1) * input_stride] = patch_result


#     return result
            

