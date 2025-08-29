import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv


class ImageAligner:
    def __init__(self, sensor_name, thermal_image_shape):
        self.sensor_name = sensor_name
        self.T_affine = None
        self.thermal_image_shape = thermal_image_shape

    def load_transformation(self, distance_range):
        with open(f"MSC/MSC_results/{self.sensor_name}{distance_range}.yaml", 'r') as file:
            data = yaml.safe_load(file)
            self.T_affine = np.array(data['T'])
        return self.T_affine
    
    def transform_image(self, image, T_affine):
        T_affine_2x3 = np.vstack([T_affine.T, [0, 0, 1]])[:2, :]
        transformed_image = cv.warpAffine(image, T_affine_2x3, (image.shape[1], image.shape[0]))
        transformed_image = transformed_image[:self.thermal_image_shape[0], :self.thermal_image_shape[1]]
        return transformed_image
    

if __name__=="__main__":
    depth = np.load("the path to the depth image")
    thermal_image_shape = (160, 120)
    it = ImageAligner("senxor_m16", thermal_image_shape)
    t_image = it.transform_image(depth, it.load_transformation("2500-3000"))
    plt.imshow(t_image)
    plt.show()
    # --------------------------------------------------------------------------------------
