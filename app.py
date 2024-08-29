import numpy as np
import cv2
import argparse
from ultralytics import YOLO

import preprocessing
from byte_tracker import BYTETracker


def main(min_confidence, nms_max_overlap,
         track_thresh, track_buffer, match_thresh,
         aspect_ratio_thresh, min_box_area, mot20,
         object_detector, video):

    COLORS = np.random.randint(0, 255, size=(200, 3),
                               dtype="uint8")

    yolo_model = YOLO(object_detector)


    tracker = BYTETracker(args)
    counter = []

    cap = cv2.VideoCapture(video)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    while (cap.isOpened()):
        ret, image = cap.read()
        if ret == True:

            results = yolo_model.predict(image)
            result = results[0]
            box_coord_list = []
            detection_list = []

            for box in result.boxes:
                if result.names[box.cls[0].item()] != 'car':
                    box_coord = box.xyxy[0].tolist()
                    # converting boxes into [x,y,w,h] (x,y)-top left corner
                    #box_coord = [box_coord[0], box_coord[1], box_coord[2] - box_coord[0], box_coord[3] - box_coord[1]]
                    box_coord = [round(x) for x in box_coord]
                    box_conf = round(box.conf[0].item(), 2)
                    box_coord.append(box_conf)
                    # box_class = result.names[box.cls[0].item()]

                    detection_list.append(box_coord)

            detections = np.array([detection_list])
            i = int(0)
            indexIDs = []
            # Update tracker.
            if detections[0] is not None:
                online_targets = tracker.update(detections[0], [image.shape[0], image.shape[1]], [image.shape[0], image.shape[1]])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tlbr = t.tlbr
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        bbox = tlbr
                        bbox = [int(b) for b in bbox]

                        indexIDs.append(int(tid))
                        counter.append(int(tid))
                        color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (color), thickness=2)
                        cv2.putText(image, str(tid), (bbox[0], bbox[1]), 0, 5e-3 * 150, (color), 2)
                        # cv2.imwrite(savefilepath+str(track.track_id)+".jpg", image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                        i += 1

                count = len(set(counter))
                cv2.putText(image, "Car Counter:" + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 0, 255), 2)

                # image_r = cv2.resize(image, (1280, 720))
                cv2.namedWindow("YOLO8_BYTEtrack", cv2.WINDOW_NORMAL)
                cv2.resizeWindow('YOLO8_BYTEtrack', 1024, 768)
                cv2.imshow("YOLO8_BYTEtrack", image)
                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            continue

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")

    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)


    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    parser.add_argument(
        "--object_detector", help="Path to  YOLOv8 Model "
                                        "", type=str, default="/home/mazhar/BYTEtrack/yolo/yolov8m.pt")
    parser.add_argument(
        "--video", help="Path to Video  "
                                        "", type=str, default='/home/mazhar/BYTEtrack/video/video.mp4')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    main(args.min_confidence, args.nms_max_overlap,
         args.track_thresh,args.track_buffer,args.match_thresh,
         args.aspect_ratio_thresh, args.min_box_area, args.mot20,
         args.object_detector, args.video)

