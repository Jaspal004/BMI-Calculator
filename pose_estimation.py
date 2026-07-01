def extract_features(image_path, debug=False):
    import cv2
    import mediapipe as mp
    from body_features import landmarks_to_features

    mp_pose = mp.solutions.pose

    # ===== READ IMAGE =====
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ===== POSE DETECTION =====
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose detected")
            return None

        features = landmarks_to_features(
            results.pose_landmarks.landmark,
            results.segmentation_mask,
            debug=debug
        )

        if features is None:
            print("Full body not visible")
            return None

        if debug:
            print("Features:", features)

        return features
