import cv2
import numpy as np
import glob
import yaml

# Konfiguracja szachownicy (9x7 wewnętrznych rogów, 20 mm bok)
CHECKERBOARD = (9, 7)
SQUARE_SIZE = 0.020  # metry

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # Rzeczywiste punkty 3D
imgpoints_left = []  # Punkty 2D lewego obiektywu
imgpoints_right = []  # Punkty 2D prawego obiektywu

images_left = sorted(glob.glob('calib_data/left/*.png'))
images_right = sorted(glob.glob('calib_data/right/*.png'))

print("Rozpoczynam detekcję narożników...")
img_shape = None

for left_path, right_path in zip(images_left, images_right):
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray_l.shape[::-1]

    # Szukanie rogów
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        y_left = corners2_l[0][0][1]
        y_right = corners2_r[0][0][1]

        if abs(y_left - y_right) > 100:
            corners2_r = corners2_r[::-1]

        imgpoints_left.append(corners2_l)
        imgpoints_right.append(corners2_r)
        print(f"Zaakceptowano parę: {left_path.split('/')[-1]}")
    else:
        print(f"Odrzucono parę (szachownica niewidoczna): {left_path.split('/')[-1]}")

if not objpoints:
    print("Błąd: Nie znaleziono poprawnych zdjęć szachownicy.")
    exit()

print("\nObliczam wewnętrzne macierze obiektywów...")
# Kalibracja każdego oka z osobna
ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

print("Wyliczam geometrię przestrzenną stereo (R & T)...")
# Obliczenie relacji przestrzennej między matrycami
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
ret_S, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K_l, D_l, K_r, D_r, img_shape, criteria=criteria_stereo, flags=flags)

# Wyliczenie macierzy rektifikacji
R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, img_shape, R, T, alpha=0)


# Zapis do standardu sensor_msgs/CameraInfo w ROS 2
def save_ros_yaml(filename, camera_name, K, D, R, P):
    calib_data = {
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": K.flatten().tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": D.flatten().tolist()},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": R.flatten().tolist()},
        "projection_matrix": {"rows": 3, "cols": 4, "data": P.flatten().tolist()}
    }
    with open(filename, 'w') as f:
        yaml.dump(calib_data, f, sort_keys=False)


save_ros_yaml("left.yaml", "narrow_stereo/left", K_l, D_l, R_l, P_l)
save_ros_yaml("right.yaml", "narrow_stereo/right", K_r, D_r, R_r, P_r)

print(f"\n[SUKCES] Błąd re-projekcji RMS: {ret_S:.4f}")
print("Zapisano gotowe pliki: left.yaml oraz right.yaml")
