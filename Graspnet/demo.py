import os
import sys
import shutil
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import torch
import matplotlib.pyplot as plt
from graspnetAPI import GraspGroup

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import io
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model
yolo_model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s', pretrained=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

def setup_google_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    drive_service = build('drive', 'v3', credentials=creds)
    return drive_service

def get_file_id_by_name(service, folder_id, file_name):
    query = f"name='{file_name}' and parents in '{folder_id}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    if len(files) > 0:
        return files[0]['id']
    else:
        return None

def download_file_from_drive(service, file_id, destination):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(destination, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

def replace_files(drive_service):
    target_dir = '/home/kyhb/catkin_ws/data'
    folder_id = '17spYQEf3v3qTsQI0D3_YSJ5nZ0FjszHW'  # Replace with your Google Drive folder ID

    file_names = ['color.png', 'depth.png']

    for file_name in file_names:
        file_id = get_file_id_by_name(drive_service, folder_id, file_name)
        if file_id:
            destination = os.path.join(target_dir, file_name)
            download_file_from_drive(drive_service, file_id, destination)
            print(f"Downloaded {file_name} from Google Drive.")
        else:
            print(f"File {file_name} not found in Google Drive.")

def upload_file_to_drive(service, file_path, parent_folder_id):
    file_name = os.path.basename(file_path)
    file_id = get_file_id_by_name(service, parent_folder_id, file_name)

    media = MediaIoBaseUpload(io.FileIO(file_path, 'rb'), mimetype='text/plain')

    if file_id:
        # Update the existing file
        file = service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f"Updated {file_name} in Google Drive with ID: {file.get('id')}")
    else:
        # Create a new file
        file_metadata = {
            'name': file_name,
            'parents': [parent_folder_id]
        }
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"Uploaded {file_name} to Google Drive with ID: {file.get('id')}")

def get_net():
    # Initialize the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # Set model to eval mode
    net.eval()
    return net

def perform_detection(image_path):
    # Load image
    image = Image.open(image_path)
    image = np.array(image)
    
    # Perform inference
    results = yolo_model(image)
    objects = results.pandas().xyxy[0]  # Results in Pandas DataFrame format
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    for index, row in objects.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(xmin, ymin, f"{index}: {row['name']}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title("Detected Objects")
    plt.axis('off')
    output_file = '/home/kyhb/catkin_ws/data/yolo_detection_result.png'
    plt.savefig(output_file)
    plt.close()
    logger.info(f"YOLO detection results saved as '{output_file}'.")
    
    # Return detected objects
    return objects

def pixel_to_camera_coords_xy(x, y, depth, intrinsic, factor_depth):
    Z = depth[y, x] / factor_depth
    X = (x - intrinsic[0][2]) * Z / intrinsic[0][0]
    Y = (y - intrinsic[1][2]) * Z / intrinsic[1][1]
    return X, Y

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    print(factor_depth)

    # generate cloud
    intrinsic[0][0] = 636.4911
    intrinsic[1][1] = 636.4911
    intrinsic[0][2] = 642.3791
    intrinsic[1][2] = 357.4644
    factor_depth = 1000
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = workspace_mask & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud, depth, intrinsic, factor_depth

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, drive_service, object_name):
    # 检查抓取组是否有有效抓取
    if len(gg) == 0:
        logger.info(f"No valid grasps found for {object_name}, skipping.")
        return  # 如果没有抓取，跳过这个物体
    
    # 否则，处理并写入有效抓取
    gg.nms()  # 进行非极大值抑制
    gg.sort_by_score()  # 按得分排序抓取
    top_grasp = gg[0]  # 获取得分最高的抓取（第一名）
    
    # 将顶级抓取转换为字符串格式
    top_grasp_str = str(top_grasp)
    
    # 写入文件，记录物体名称和其对应的最高抓取位姿
    gg_file_path = '/home/kyhb/catkin_ws/data/gg_values.txt'
    with open(gg_file_path, 'a') as file:  # 使用追加模式记录多个物体
        file.write(f"Object Name: {object_name}\n")  # 写入物体名称
        file.write("Top Grasp Value:\n")
        file.write(top_grasp_str)  # 只写入最高抓取
        file.write("\n\n")  # 用空行分隔每个物体的抓取位姿
    
    # 上传文件到Google Drive
    parent_folder_id = '17spYQEf3v3qTsQI0D3_YSJ5nZ0FjszHW'  # 替换为你的Google Drive文件夹ID
    upload_file_to_drive(drive_service, gg_file_path, parent_folder_id)






def demo(data_dir):
    drive_service = setup_google_drive()  # 初始化Google Drive API
    
    # 清空文件中的前一轮结果
    gg_file_path = '/home/kyhb/catkin_ws/data/gg_values.txt'
    with open(gg_file_path, 'w') as file:
        file.write("")  # 在运行开始时清空文件
    
    processed_objects = set()  # 用来跟踪已经处理过的物体
    
    try:
        while True:
            replace_files(drive_service)  # 在开始演示前更新文件
            
            # 步骤1：执行物体检测
            objects = perform_detection(os.path.join(data_dir, 'color.png'))
            if len(objects) == 0:
                logger.info("No objects detected. Retrying...")
                time.sleep(5)
                continue
            
            # 步骤2：处理数据并生成抓取
            net = get_net()
            end_points, cloud, depth, intrinsic, factor_depth = get_and_process_data(data_dir)
            
            # 步骤3：针对每个检测到的物体进行处理
            for index, obj in objects.iterrows():
                object_name = obj['name']  # 获取物体名称

                # 检查物体是否已经处理过
                if object_name in processed_objects:
                    logger.info(f"Object '{object_name}' already processed, skipping.")
                    continue  # 如果已经处理过，跳过这个物体
                
                xmin, ymin, xmax, ymax = map(int, [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
                
                # 基于3D空间中的边界框过滤抓取
                xmin_3d_x, xmin_3d_y = pixel_to_camera_coords_xy(xmin, ymin, depth, intrinsic, factor_depth)
                xmax_3d_x, xmax_3d_y = pixel_to_camera_coords_xy(xmax, ymax, depth, intrinsic, factor_depth)

                gg = get_grasps(net, end_points)

                filtered_gg = GraspGroup()
                for grasp in gg:
                    translation = grasp.translation
                    if (xmin_3d_x <= translation[0] <= xmax_3d_x and
                        xmin_3d_y <= translation[1] <= xmax_3d_y):
                        filtered_gg.add(grasp)

                # 如果有必要，进行碰撞检测
                if cfgs.collision_thresh > 0:
                    filtered_gg = collision_detection(filtered_gg, np.array(cloud.points))

                # 步骤4：只写入每个物体的最高抓取结果
                vis_grasps(filtered_gg, drive_service, object_name)

                # 记录该物体已经处理过
                processed_objects.add(object_name)

            # 每5秒更新一次
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
        print("Demo interrupted by user.")




if __name__ == '__main__':
    data_dir = '/home/kyhb/catkin_ws/data'
    demo(data_dir)