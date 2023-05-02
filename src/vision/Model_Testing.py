import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Model_loader import YOLOv5Model
class ModelTesting:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def preprocess_image(self, image):
        # TODO: Implement image processing/augmentation techniques
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        return blob

    def postprocess_output(self, output, image):
        # Compute object detections
        detections = output[0, 0, :, :]
        h, w = image.shape[:2]
        
        boxes = []
        centroids = []

        for detection in detections:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                centroid = [centerX, centerY]
                boxes.append(box)
                centroids.append(centroid)
        return boxes, centroids

    def evaluate(self, boxes, centroids, correct_boxes, correct_centroids):
        # Compute classification accuracy
        correct_classifications = sum([box in correct_boxes for box in boxes])
        classification_accuracy = correct_classifications / len(correct_boxes)

        # Compute task success rate
        correct_order = centroids[0][1] < centroids[1][1] if len(centroids) == 2 else False
        task_success = classification_accuracy == 1 and correct_order

        return classification_accuracy, task_success
        
    def test_model(self, test_loader, correct_outputs):
        self.model.eval()
        classification_accuracies = []
        task_successes = []
        with torch.no_grad():
            for i, image in enumerate(test_loader):
                image = self.preprocess_image(image)
                output = self.model(image)
                boxes, centroids = self.postprocess_output(output, image)
                
                correct_boxes, correct_centroids = correct_outputs[i]
                classification_accuracy, task_success = self.evaluate(
                    boxes, centroids, correct_boxes, correct_centroids)
                classification_accuracies.append(classification_accuracy)
                task_successes.append(task_success)

        average_classification_accuracy = np.mean(classification_accuracies)
        task_success_rate = np.mean(task_successes)

        print(f'Average Classification Accuracy: {average_classification_accuracy}')
        print(f'Task Success Rate: {task_success_rate}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolov5_model = YOLOv5Model('yolov5s.pt')
    # similarly create instances for other models...

    image = cv2.imread('image.jpg')
    boxes, centroids = yolov5_model.infer(image)
    # use boxes and centroids for downstream tasks...

    # TODO: Replace with chosen model
    model = None
    model = model.to(device)
    tester = ModelTesting(model, device)

    # TODO: Load test images
    test_images = []

    tester.test_model(test_images)

if __name__ == '__main__':
    main()
