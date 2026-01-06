import os
import io
import base64

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.augmentations.functional import image_compression
from facenet_pytorch.models.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor

from torchvision.transforms import Normalize

np.int = int  # temporary patch for deprecated alias

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

def get_device():
    """Helper to get the most appropriate available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Global device variable
DEVICE = get_device()



class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, verbose=True, insets=(0, 0)):
        """Creates a new VideoReader.

        Arguments:
            verbose: whether to print warnings and error messages
            insets: amount to inset the image by, as a percentage of
                (width, height). This lets you "zoom in" to an image
                to remove unimportant content around the borders.
                Useful for face detection, which may not work if the
                faces are too small.
        """
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path, num_frames, jitter=0, seed=None):
        """Reads frames that are always evenly spaced throughout the video.

        Arguments:
            path: the video file
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
            jitter: if not 0, adds small random offsets to the frame indices;
                this is useful so we don't always land on even or odd frames
            seed: random seed for jittering; if you set this to a fixed value,
                you probably want to set it only on the first video
        """
        assert num_frames > 0

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def read_random_frames(self, path, num_frames, seed=None):
        """Picks the frame indices at random.

        Arguments:
            path: the video file
            num_frames: how many frames to read, -1 means the entire video
                (warning: this will take up a lot of memory!)
        """
        assert num_frames > 0
        np.random.seed(seed)

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None

        frame_idxs = sorted(np.random.choice(np.arange(0, frame_count), num_frames))
        result = self._read_frames_at_indices(path, capture, frame_idxs)

        capture.release()
        return result

    def read_frames_at_indices(self, path, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array.

        Arguments:
            path: the video file
            frame_idxs: a list of frame indices. Important: should be
                sorted from low-to-high! If an index appears multiple
                times, the frame is still read only once.

        Returns:
            - a NumPy array of shape (num_frames, height, width, 3)
            - a list of the frame indices that were read

        Reading stops if loading a frame fails, in which case the first
        dimension returned may actually be less than num_frames.

        Returns None if an exception is thrown for any reason, or if no
        frames were read.
        """
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(path)
        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                # Get the next frame, but don't decode if we're not using it.
                ret = capture.grab()
                if not ret:
                    if self.verbose:
                        print("Error grabbing frame %d from movie %s" % (frame_idx, path))
                    break

                # Need to look at this frame?
                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                        break

                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)

            if len(frames) > 0:
                return np.stack(frames), idxs_read
            if self.verbose:
                print("No frames read from movie %s" % path)
            return None
        except:
            if self.verbose:
                print("Exception while reading movie %s" % path)
            return None

    def read_middle_frame(self, path):
        """Reads the frame from the middle of the video."""
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        result = self._read_frame_at_index(path, capture, frame_count // 2)
        capture.release()
        return result

    def read_frame_at_index(self, path, frame_idx):
        """Reads a single frame from a video.

        If you just want to read a single frame from the video, this is more
        efficient than scanning through the video to find the frame. However,
        for reading multiple frames it's not efficient.

        My guess is that a "streaming" approach is more efficient than a
        "random access" approach because, unless you happen to grab a keyframe,
        the decoder still needs to read all the previous frames in order to
        reconstruct the one you're asking for.

        Returns a NumPy array of shape (1, H, W, 3) and the index of the frame,
        or None if reading failed.
        """
        capture = cv2.VideoCapture(path)
        result = self._read_frame_at_index(path, capture, frame_idx)
        capture.release()
        return result

    def _read_frame_at_index(self, path, capture, frame_idx):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret or frame is None:
            if self.verbose:
                print("Error retrieving frame %d from movie %s" % (frame_idx, path))
            return None
        else:
            frame = self._postprocess_frame(frame)
            return np.expand_dims(frame, axis=0), [frame_idx]

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]

        if self.insets[1] > 0:
            H = frame.shape[1]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]

        return frame


class FaceExtractor:
    def __init__(self, video_read_fn):
        self.video_read_fn = video_read_fn
        # Use CPU for MTCNN on Mac to avoid some MPS specific ops issues, or try MPS if stable
        # Using device=DEVICE might fail if MPS doesn't support some MTCNN ops, fallback to cpu for safety if MPS
        mtcnn_device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=mtcnn_device)

    def process_videos(self, input_dir, filenames, video_idxs):
        videos_read = []
        frames_read = []
        frames = []
        results = []
        for video_idx in video_idxs:
            # Read the full-size frames from this video.
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)
            # Error? Then skip this video.
            if result is None: continue

            videos_read.append(video_idx)

            # Keep track of the original frames (need them later).
            my_frames, my_idxs = result

            frames.append(my_frames)
            frames_read.append(my_idxs)
            for i, frame in enumerate(my_frames):
                h, w = frame.shape[:2]
                img = Image.fromarray(frame.astype(np.uint8))
                img = img.resize(size=[s // 2 for s in img.size])

                batch_boxes, probs = self.detector.detect(img, landmarks=False)

                faces = []
                scores = []
                if batch_boxes is None:
                    continue
                for bbox, score in zip(batch_boxes, probs):
                    if bbox is not None:
                        xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                        w = xmax - xmin
                        h = ymax - ymin
                        p_h = h // 3
                        p_w = w // 3
                        crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                        faces.append(crop)
                        scores.append(score)

                frame_dict = {"video_idx": video_idx,
                              "frame_idx": my_idxs[i],
                              "frame_w": w,
                              "frame_h": h,
                              "faces": faces,
                              "scores": scores}
                results.append(frame_dict)

        return results

    def process_video(self, video_path):
        """Convenience method for doing face extraction on a single video."""
        input_dir = os.path.dirname(video_path)
        filenames = [os.path.basename(video_path)]
        return self.process_videos(input_dir, filenames, [0])


def detect_faces_in_image(img_rgb, detector, padding_ratio=0.33):
    """
    Detect and extract faces from an image.
    
    Arguments:
        img_rgb: numpy array of the image in RGB format (H x W x 3)
        detector: MTCNN detector instance
        padding_ratio: padding around face as ratio of face size (default: 0.33 = 33%)
    
    Returns:
        list of dicts, each containing:
            - bbox: dict with xmin, ymin, xmax, ymax
            - confidence: detection confidence score
            - face_crop: numpy array of the cropped face (RGB)
    """
    img_pil = Image.fromarray(img_rgb.astype(np.uint8))
    
    # Detect faces
    boxes, probs = detector.detect(img_pil, landmarks=False)
    
    faces = []
    if boxes is not None:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if box is not None:
                xmin, ymin, xmax, ymax = [int(b) for b in box]
                
                # Add padding to face crop
                w = xmax - xmin
                h = ymax - ymin
                p_h = int(h * padding_ratio)
                p_w = int(w * padding_ratio)
                
                # Crop face with padding (clamped to image bounds)
                crop_ymin = max(ymin - p_h, 0)
                crop_ymax = min(ymax + p_h, img_rgb.shape[0])
                crop_xmin = max(xmin - p_w, 0)
                crop_xmax = min(xmax + p_w, img_rgb.shape[1])
                
                face_crop = img_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                
                faces.append({
                    'face_index': i,
                    'bbox': {
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    },
                    'confidence': float(prob),
                    'face_crop': face_crop
                })
    
    return faces



def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

strategy = confident_strategy


def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def predict_on_video(face_extractor, video_path, batch_size, input_size, models, strategy=np.mean,
                     apply_compression=False):
    batch_size *= 4
    try:
        faces = face_extractor.process_video(video_path)
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = put_to_center(resized_face, input_size)
                    if apply_compression:
                        resized_face = image_compression(resized_face, quality=90, image_type=".jpg")
                    if n + 1 < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        pass
            if n > 0:
                x = torch.tensor(x, device=DEVICE).float()
                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                # Make a prediction, then take the average.
                with torch.no_grad():
                    preds = []
                    for model in models:
                        # Handle mixed precision (half) only if CUDA is available, otherwise float32
                        input_tensor = x[:n].half() if DEVICE == "cuda" else x[:n].float()
                        y_pred = model(input_tensor)
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        bpred = y_pred[:n].cpu().numpy()
                        preds.append(strategy(bpred))
                    return np.mean(preds)
    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


def predict_on_video_set(face_extractor, videos, input_size, num_workers, test_dir, frames_per_video, models,
                         strategy=np.mean,
                         apply_compression=False):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(face_extractor=face_extractor, video_path=os.path.join(test_dir, filename),
                                  input_size=input_size,
                                  batch_size=frames_per_video,
                                  models=models, strategy=strategy, apply_compression=apply_compression)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))
    return list(predictions)


def generate_gradcam(model, input_tensor, target_layer=None):
    """
    Generate Grad-CAM heatmap for a specific input and model.
    """
    if target_layer is None:
        # Default to the last convolutional layer of the encoder (EfficientNet)
        # Structure depends on timm/efficientnet implementation
        # Usually model.encoder.conv_head or model.encoder.blocks[-1]
        try:
            target_layer = model.encoder.conv_head
        except:
            return None

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)

    # Register hooks
    handle_grad = target_layer.register_backward_hook(lambda m, i, o: save_gradient(o[0]))
    handle_act = target_layer.register_forward_hook(save_activation)

    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    output = torch.sigmoid(output)
    
    # Backward pass with respect to the output class (fake)
    output.backward()

    # Generate heatmap
    if gradients and activations:
        grad = gradients[0].cpu().data.numpy()[0]
        act = activations[0].cpu().data.numpy()[0]
        
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)  # Normalize
        
        # Cleanup hooks
        handle_grad.remove()
        handle_act.remove()
        
        return cam, output.item()
    
    handle_grad.remove()
    handle_act.remove()
    return None, output.item()


def predict_on_image(face_crop, input_size, models, strategy=np.mean, with_heatmap=False):
    """
    Predict deepfake probability on a single face crop.
    
    Arguments:
        face_crop: numpy array of the face image (RGB format, H x W x 3)
        input_size: target size for model input (e.g., 380)
        models: list of loaded DeepFakeClassifier models
        strategy: aggregation strategy for model predictions (default: np.mean)
        with_heatmap: boolean, if True returns (prediction, heatmap_base64)
    
    Returns:
        float: prediction score (0 = real, 1 = fake)
        OR
        (float, str): prediction score and base64 heatmap (if with_heatmap=True)
    """
    try:
        # Preprocess face for model
        face_resized = isotropically_resize_image(face_crop, input_size)
        face_centered = put_to_center(face_resized, input_size)
        
        # Convert to tensor
        face_tensor = torch.tensor(face_centered).float().permute(2, 0, 1) / 255.0
        face_tensor = normalize_transform(face_tensor)
        
        # Prepare input tensor
        if DEVICE == "cuda":
            face_tensor = face_tensor.unsqueeze(0).to(DEVICE).half()
        else:
            face_tensor = face_tensor.unsqueeze(0).to(DEVICE).float()
        
        # Get predictions from all models
        preds = []
        heatmap = None
        
        # For prediction, we use no_grad unless we need heatmap
        context = torch.enable_grad() if with_heatmap else torch.no_grad()
        
        with context:
            for i, model in enumerate(models):
                if with_heatmap and i == 0: # Generate heatmap only from the first model
                     # Make sure model requires grad for heatmap
                     # We might need to temporarily set mode to train or allow gradients? 
                     # Actually eval mode is fine for grad-cam usually if params require grad
                     # But loaded models might have requires_grad=False
                     
                     CAM, pred_val = generate_gradcam(model, face_tensor)
                     if CAM is not None:
                         # Merge heatmap with original image
                         heatmap_img = cv2.applyColorMap(np.uint8(255 * CAM), cv2.COLORMAP_JET)
                         heatmap_img = np.float32(heatmap_img) / 255
                         
                         # Resize original face to input size for overlay
                         overlay_img = cv2.cvtColor(face_centered, cv2.COLOR_RGB2BGR)
                         overlay_img = np.float32(overlay_img) / 255
                         
                         # Blend
                         cam_result = heatmap_img + overlay_img
                         cam_result = cam_result / np.max(cam_result)
                         cam_result = np.uint8(255 * cam_result)
                         cam_result = cv2.cvtColor(cam_result, cv2.COLOR_BGR2RGB)
                         
                         # Encode to base64
                         pil_img = Image.fromarray(cam_result)
                         buff = io.BytesIO()
                         pil_img.save(buff, format="JPEG")
                         heatmap = base64.b64encode(buff.getvalue()).decode("utf-8")
                         
                     preds.append(pred_val)
                else:
                    with torch.no_grad():
                        y_pred = model(face_tensor)
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        preds.append(y_pred.cpu().numpy().item())
        
        # Aggregate predictions
        result = strategy(preds)
        
        if with_heatmap:
            return result, heatmap
        return result
        
    except Exception as e:
        print(f"Prediction error on image: {str(e)}")
        if with_heatmap:
            return 0.5, None
        return 0.5
