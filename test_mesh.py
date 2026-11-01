import cv2, mediapipe as mp

img = cv2.imread("face.jpg")
mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print("face found:", res.multi_face_landmarks is not None)

h, w = img.shape[:2]
for lm in res.multi_face_landmarks[0].landmark:
    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
cv2.imwrite("mesh.jpg", img)
