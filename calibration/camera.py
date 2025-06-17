import pyzed.sl as sl  
import cv2  
import numpy as np  

class Camera:  
    def __init__(self):  
        self.zed = sl.Camera()  
        self.init_params = sl.InitParameters()  
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  
        self.init_params.camera_fps = 30  
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL  
        self.init_params.coordinate_units = sl.UNIT.METER  
        
        # Intrinsics matrix for left camera  
        self.color_intrinsics_mat = np.array([  
            [1065.26, 0, 936.44],  
            [0, 1065.32, 502.622],  
            [0, 0, 1]  
           
        ], dtype=np.float64)  


        self.opened = False  
        self.image = sl.Mat()  
        self.depth = sl.Mat()  

    def start(self):  
        try:  
            err = self.zed.open(self.init_params)  
            if err != sl.ERROR_CODE.SUCCESS:  
                raise ValueError(f"ZED Camera open failed: {err}")  
            self.opened = True  
            print("Camera started successfully.")  
        except Exception as e:  
            print(f"Failed to start camera: {str(e)}")  
            raise  

    def get_image(self):  
        if not self.opened:  
            return None, None  

        try:  
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:  
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)  
                image_np = self.image.get_data()  

                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)  
                depth_np = self.depth.get_data()  

                return image_np, depth_np  
            else:  
                print("Failed to grab frame from camera.")  
                return None, None  
        except Exception as e:  
            print(f"Error getting camera image: {str(e)}")  
            return None, None  

    def stop(self):  
        try:  
            if self.opened:  
                self.zed.close()  
                cv2.destroyAllWindows()  
                self.opened = False  
                print("Camera stopped successfully.")  
        except Exception as e:  
            print(f"Error stopping camera: {str(e)}")  
    
    def display_images(self):  
        while self.opened:  
            image_np, depth_np = self.get_image()  
            if image_np is None or depth_np is None:  
                continue  
            
            # Display RGB image  
            cv2.imshow("RGB Image", image_np)  

            # Normalize depth for visualization  
            depth_display = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)  
            depth_display = depth_display.astype(np.uint8)  # Convert to uint8  
            
            # Display depth image  
            cv2.imshow("Depth Image", depth_display)  

            # Exit on 'q' key press  
            key = cv2.waitKey(1)  
            if key == ord('q'):  
                break  

if __name__ == "__main__":  
    camera = Camera()  
    camera.start()  
    try:  
        camera.display_images()  
    finally:  
        camera.stop()