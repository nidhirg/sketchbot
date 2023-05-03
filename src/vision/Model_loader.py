# Abstract base class for all models
class AbstractModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        raise NotImplementedError

    def preprocess_image(self, image):
        raise NotImplementedError

    def postprocess_output(self, output):
        raise NotImplementedError

    def infer(self, image):
        preprocessed_image = self.preprocess_image(image)
        output = self.model(preprocessed_image)
        return self.postprocess_output(output)


# Example implementation for YOLOv5
from yolov5 import LoadModel

class YOLOv5Model(AbstractModel):
    def load_model(self, model_path):
        return LoadModel(model_path)

    def preprocess_image(self, image):
        img = cv2.resize(image, (416, 416))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img.unsqueeze(0)

    def postprocess_output(self, output):
        boxes = output.xyxy[0].cpu().numpy()
        centroids = [[(box[0]+box[2])/2, (box[1]+box[3])/2] for box in boxes]
        return boxes, centroids

