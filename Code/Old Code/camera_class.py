class Camera:
    def __init__(self, user, video_folder, file_name):
        self.user = user
        self.video_folder = video_folder
        self.file_name = file_name

        self.parent_path = f"/Users/{self.user}/Library/Mobile Documents/com~apple~CloudDocs/Bachelor Project/"
        self.parent_path_video = f"{self.parent_path}Videos/{self.video_folder}/"
        self.tracker_path = f"{self.parent_path_video}Data/{self.file_name}/tracker_{self.file_name}.json"
        self.video_path = f"{self.parent_path_video}Processed/{self.file_name}.mp4"
        self.photo_path = f"{self.parent_path_video}Photos/{self.file_name}"
        self.base_image = f"{self.parent_path}Base Image"
        self.map_path = f"{self.base_image}/FullHD_bridge.png"

if __name__ == "__main__":
    pass