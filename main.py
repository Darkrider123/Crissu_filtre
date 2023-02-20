import cv2
import numpy as np

def filter_image(image, kernel):
    image = np.array(image, np.uint8)
    filtered_image = cv2.filter2D(image, cv2.CV_8U, kernel)
    return filtered_image


def Canny(image, threshold_strong_lines, threshold_weak_lines = None, sobel_kernerl_size = 3):
    if threshold_strong_lines is None:
        threshold_weak_lines = threshold_strong_lines / 2

    image = cv2.Canny(image, threshold_weak_lines, threshold_strong_lines, None, apertureSize=sobel_kernerl_size)
    return image


def Sobel(image, pa_orizontala, pa_verticala, kernel_size):
    return cv2.Sobel(image, cv2.CV_8U, dx=pa_orizontala, dy=pa_verticala, ksize=kernel_size)


def main():
    image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)
    image_Sobel_o = Sobel(image, 1, 0, 3)
    image_Sobel_v = Sobel(image, 0, 1, 3)
    image_Sobel_both = Sobel(image, 1, 1, 3)
    image_Canny = Canny(image, 600) #Canny e grayscale
    image_Canny = cv2.cvtColor(image_Canny, cv2.COLOR_GRAY2BGR) #Il facem BGR(asa merge cv2, nu pe RGB)
    image_custom_sa_zicem_ca_vrem_feature_uri_pa_diagonala_principala = filter_image(image, 
    np.array([  [-2, -1, 0,],
                [-1, 0, 1,],
                [0, 1, 2]
                ]))# E un fel de Sobel, dar pe diagonala

    image_combi = np.concatenate([image, image_Sobel_o, image_Sobel_v, image_Sobel_both, image_Canny, image_custom_sa_zicem_ca_vrem_feature_uri_pa_diagonala_principala], axis=1)
    image_combi = cv2.resize(image_combi, (1800, 500)) #Resize sa incapa frumi pe ecran

    cv2.imshow("Original/Sobel_orizonatala/Sobel_verticala/Sobel_ambele/Canny/feature_custom", image_combi)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
