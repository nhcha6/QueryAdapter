
from conceptgraph.dataset.datasets_common import *

class ScannetppDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, 'data', sequence)
        self.pose_path = os.path.join(self.input_folder, "iphone/colmap/images.txt")

        # load the camera intrinsics, image height and width into the config_dict
        self.intrinsic_path = os.path.join(self.input_folder, 'iphone/colmap/cameras.txt')

        with open(self.intrinsic_path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line[0] == '#':
                continue
            else:
                camera_details = line.split()
                break
        
        config_dict['camera_params']['fx'] = camera_details[4]
        config_dict['camera_params']['fy'] = camera_details[5]
        config_dict['camera_params']['cx'] = camera_details[6]
        config_dict['camera_params']['cy'] = camera_details[7]


        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        # the original replica paths
        # color_paths = natsorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        # depth_paths = natsorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        
        # the scannetpp
        # pose_path = '/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/data/0a5c013435/iphone/colmap/images.txt'
        posed_imgs = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            try:
                int(line[0])
            except:
                continue
            line_split = line.split()
            posed_imgs.append(line_split[-1])
        
        # input_folder = '/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/data/0a5c013435'
        color_paths = natsorted(glob.glob(f"{self.input_folder}/iphone/rgb/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/iphone/depth/*.png"))

        # only keep the images that have poses
        color_paths = [color_paths[i] for i in range(len(color_paths)) if color_paths[i].split('/')[-1] in posed_imgs]
        depth_paths = [depth_paths[i] for i in range(len(depth_paths)) if depth_paths[i].split('/')[-1].replace('png', 'jpg') in posed_imgs]

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        # poses = []
        # with open(self.pose_path, "r") as f:
        #     lines = f.readlines()
        # print(lines)
        # for i in range(self.num_imgs):
        #     line = lines[i]
        #     c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        #     # c2w[:3, 1] *= -1
        #     # c2w[:3, 2] *= -1
        #     c2w = torch.from_numpy(c2w).float()
        #     poses.append(c2w)

        # pose_path = '/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/data/0a5c013435/iphone/colmap/images.txt'

        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            try:
                int(line[0])
            except:
                continue
            line_split = line.split()

            # generate transformation matrix from quaternion and translation
            q = np.array(list(map(float, line_split[1:5])))
            t = np.array(list(map(float, line_split[5:8])))
            c2w = np.eye(4)
            Rot = R.from_quat(q, scalar_first=True).as_matrix()

            # switch y and z axis
            # Rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ Rot
            # t = np.array([t[0], t[2], t[1]])

            c2w[:3, :3] = Rot.transpose()
            c2w[:3, 3] = -Rot.transpose().dot(t)

            # c2w[:3, :3] = Rot
            # c2w[:3, 3] = t

            # c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # # c2w[:3, 1] *= -1
            # # c2w[:3, 2] *= -1

            c2w = torch.from_numpy(c2w).float()

            poses.append(c2w)
        

        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

def get_scannet_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    return ScannetppDataset(config_dict, basedir, sequence, **kwargs)