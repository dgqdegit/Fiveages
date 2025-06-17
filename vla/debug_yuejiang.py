import os
import sys
import numpy as np
from frankapy import FrankaArm
sys.path.append("/media/casia/data2/lpy/3D_VLA/code/RVT")
from data_collection_frankapy import osc_move_eval,gripper_open_flag
from utils.real_camera_utils_new import Camera,get_cam_extrinsic
import torch
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import rvt_our.mvt.config as default_mvt_cfg
import rvt_our.models.rvt_agent as rvt_agent
import rvt_our.config as default_exp_cfg

from rvt_our.utils.rvt_utils import load_agent as load_agent_state

from rvt_our.mvt.mvt import MVT
from rvt_our.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)




def convert_pcd_to_base(
            type="3rd",
            pcd=[]
        ):
        transform = get_cam_extrinsic(type)
        
        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3)
        
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
        pcd = (transform @ pcd.T).T[:, :3]
        
        pcd = pcd.reshape(h, w, 3)
        return pcd 



def load_agent(
    model_path=None,
    peract_official=False,
    peract_model_dir=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    device=0,
    use_input_place_with_mean=False,
    palligemma_type=1,
    layer_index=-1,
    layer_concat=False):
    device = f"cuda:{device}"
    if not (peract_official):
        assert model_path is not None

        # load exp_cfg
        model_folder = os.path.join(os.path.dirname(model_path))

        exp_cfg = default_exp_cfg.get_cfg_defaults()
        if exp_cfg_path != None:
            exp_cfg.merge_from_file(exp_cfg_path)
        else:
            exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

        # NOTE: to not use place_with_mean in evaluation
        # needed for rvt-1 but not rvt-2
        if not use_input_place_with_mean:
            # for backward compatibility
            old_place_with_mean = exp_cfg.rvt.place_with_mean
            exp_cfg.rvt.place_with_mean = True
        exp_cfg.freeze()

        # create agent
        if exp_cfg.agent == "original":
            assert False
            # initialize PerceiverIO Transformer
            VOXEL_SIZES = [100]  # 100x100x100 voxels
            NUM_LATENTS = 512  # PerceiverIO latents
            BATCH_SIZE_TRAIN = 1
            perceiver_encoder = PerceiverIO(
                depth=6,
                iterations=1,
                voxel_size=VOXEL_SIZES[0],
                initial_dim=3 + 3 + 1 + 3,
                low_dim_size=4,
                layer=0,
                num_rotation_classes=72,
                num_grip_classes=2,
                num_collision_classes=2,
                num_latents=NUM_LATENTS,
                latent_dim=512,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                weight_tie_layers=False,
                activation="lrelu",
                input_dropout=0.1,
                attn_dropout=0.1,
                decoder_dropout=0.0,
                voxel_patch_size=5,
                voxel_patch_stride=5,
                final_dim=64,
            )

            # initialize PerceiverActor
            agent = PerceiverActorAgent(
                coordinate_bounds=SCENE_BOUNDS,
                perceiver_encoder=perceiver_encoder,
                camera_names=CAMERAS,
                batch_size=BATCH_SIZE_TRAIN,
                voxel_size=VOXEL_SIZES[0],
                voxel_feature_size=3,
                num_rotation_classes=72,
                rotation_resolution=5,
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                transform_augmentation=False,
                **exp_cfg.peract,
            )
        elif exp_cfg.agent == "our":
            mvt_cfg = default_mvt_cfg.get_cfg_defaults()
            if mvt_cfg_path != None:
                mvt_cfg.merge_from_file(mvt_cfg_path)
            else:
                mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
            mvt_cfg.add_proprio=False
            mvt_cfg.freeze()

            # for rvt-2 we do not change place_with_mean regardless of the arg
            # done this way to ensure backward compatibility and allow the
            # flexibility for rvt-1
            if mvt_cfg.stage_two:
                exp_cfg.defrost()
                exp_cfg.rvt.place_with_mean = old_place_with_mean
                exp_cfg.freeze()

            rvt = MVT(
                renderer_device=device,
                palligemma_type=palligemma_type,
                layer_concat=layer_concat,
                layer_index=layer_index,
                **mvt_cfg,
            )

            agent = rvt_agent.RVTAgent(
                network=rvt.to(device),
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                add_lang=mvt_cfg.add_lang,
                stage_two=mvt_cfg.stage_two,
                rot_ver=mvt_cfg.rot_ver,
                scene_bounds=SCENE_BOUNDS,
                cameras=CAMERAS,
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
        else:
            raise NotImplementedError

        agent.build(training=False, device=device)
        load_agent_state(model_path, agent)
        agent.eval()

    elif peract_official:  # load official peract model, using the provided code
        assert False
        try:
            model_folder = os.path.join(os.path.abspath(peract_model_dir), "..", "..")
            train_cfg_path = os.path.join(model_folder, "config.yaml")
            agent = get_official_peract(train_cfg_path, False, device, bs=1)
        except FileNotFoundError:
            print("Config file not found, trying to load again in our format")
            train_cfg_path = "configs/peract_official_config.yaml"
            agent = get_official_peract(train_cfg_path, False, device, bs=1)
        agent.load_weights(peract_model_dir)
        agent.eval()

    print("Agent Information")
    print(agent)
    return agent



import threading
import queue
import cv2
from collections import deque
from datetime import datetime

import os
import shutil
import sys

def check_and_prepare_directory(target_path):
    """
    检查并准备目标路径：
    1. 如果路径存在且非空，询问用户是否删除内容
    2. 如果路径不存在，创建目录
    3. 处理权限错误等异常情况
    """
    try:
        # 检查路径是否存在
        if os.path.exists(target_path):
            # 列出目录内容（不包含子目录中的文件）
            dir_contents = os.listdir(target_path)
            
            if dir_contents:  # 如果目录非空
                print(f"目录 {target_path} 已存在且包含内容")
                choice = input("是否删除目录下所有内容？(y/n): ").lower()
                
                if choice == 'y':
                    # 递归删除目录下所有内容
                    for filename in dir_contents:
                        file_path = os.path.join(target_path, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)  # 删除文件或符号链接
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # 递归删除子目录
                    print("已清空目录")
                elif choice == 'n':
                    print("操作已取消")
                    sys.exit(0)
                else:
                    print("无效输入，操作已取消")
                    sys.exit(1)
            else:
                print(f"目录 {target_path} 已存在且为空")
        else:
            # 创建目录（包括中间目录）
            os.makedirs(target_path, exist_ok=True)
            print(f"已创建目录：{target_path}")
            
        return True
    
    except PermissionError as pe:
        print(f"权限不足：{str(pe)}")
        sys.exit(1)
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
        sys.exit(1)





class ImageCaptureAndSaver(threading.Thread):
    def __init__(self, cameras, save_path, frequency=30):
        super().__init__()
        self.cameras = cameras
        self.save_path = save_path
        self.frequency = frequency  # 每秒保存图像的频率
        self.lock = threading.Lock()
        self.latest_frame = {
            'timestamp': None, 
            'rgb': None,  # 统一键名
            'pcd': None
        }
        self.wait_time = 1 / frequency  # 控制捕捉图像的频率
        self._stop_event = threading.Event()  # 用于停止线程

    def run(self):
        last_save_time = None  # 初始化为 None
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                frame = self.cameras.capture()
                
                # 生成时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                with self.lock:
                    # 创建独立内存副本
                    new_rgb = np.empty_like(frame["3rd"]["rgb"])
                    new_pcd = np.empty_like(frame["3rd"]["pcd"])
                    np.copyto(new_rgb, frame["3rd"]["rgb"])
                    np.copyto(new_pcd, frame["3rd"]["pcd"])
                    
                    self.latest_frame = {
                        'timestamp': timestamp,  # 使用当前生成的时间戳,
                        'rgb': new_rgb,  # 使用独立内存
                        'pcd': new_pcd
                    }

                # 保存条件判断（使用时间戳字符串比较）
                if last_save_time != timestamp:  # ✅ 直接使用当前生成的 timestamp
                    self.save_image(
                        image=self.latest_frame['rgb'], 
                        timestamp=timestamp  # ✅ 传递正确的时间戳
                    )
                    last_save_time = timestamp

                # 精确频率控制
                elapsed = time.time() - start_time
                if elapsed < self.wait_time:
                    time.sleep(self.wait_time - elapsed)
                    
            except KeyError as ke:
                print(f"数据键错误: {str(ke)}")
                self._stop_event.set()
            except Exception as e:
                print(f"捕获或保存错误: {str(e)}")

    def save_image(self, image, timestamp):  # ✅ 参数名明确
        try:
            save_file = os.path.join(self.save_path, f"{timestamp}.png")
            cv2.imwrite(save_file, image)
            # print(f"图像已保存: {save_file}")
        except Exception as e:
            print(f"保存图像错误: {str(e)}")
    def get_latest(self):
        with self.lock:  # 获取锁后再拷贝
            if self.latest_frame['rgb'] is None:
                return None
            # 返回深拷贝
            return {
                'timestamp': self.latest_frame['timestamp'],
                'rgb': self.latest_frame['rgb'].copy(),
                'pcd': self.latest_frame['pcd'].copy()
            }
    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)  # 设置2秒超时
        if self.is_alive():
            print("警告：图像保存线程未能正常停止")




def _eval():
    task_class="basic"   #  change 1
    instruction="put the wolf in the upper drawer"   # change 2
    trial_id= 3 # change 3
    model_name="debug" # change 4
    # model_file_name="no_pretrain_300.pth" # change 5
    # model_file_name="layer_1_only_one_image_pretrain_13tasks_lr2e5_300.pth"
    # model_file_name="13task_layer_1_only_one_image_pretrain_300.pth"
    model_file_name="yuejiang_2.pth"
    root_save_path="/media/casia/data3/lpy_data/3D_VLA/rollout_results"
    save_path=os.path.join(root_save_path,model_name,task_class,f"trial_{trial_id}",instruction.replace(" ","_"),"imgs")
    check_and_prepare_directory(save_path)


    base_path = "/media/casia/data2/lpy/3D_VLA/code/checkpoints/final_ckpt"
    model_path = os.path.join(base_path, model_file_name)
    camera="3rd" # 目前仅支持3rd
    layer_index=-1
    exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")  
    mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")
    episode_length = 20
    gripper_thres=0.076
    cameras_view=[camera]
    device = f"cuda:0"
    observation = {}
    instructions = [[[instruction]]]
    cameras = Camera(camera_type=camera) # modify it
    time.sleep(1)
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    time.sleep(2)
    print("******Begin load checkpoint!********")
    agent = load_agent(
        model_path=model_path,
        exp_cfg_path=exp_cfg_path,
        mvt_cfg_path=mvt_cfg_path,
        device=0,
        palligemma_type=1,
        layer_index=layer_index,  # change it
        layer_concat=False,
        )
    observation["language_goal"]=instructions



    # 初始化图像捕获和保存线程
    image_capture_and_saver = ImageCaptureAndSaver(cameras, save_path, frequency=15)  
    image_capture_and_saver.start()
    # 在 _eval() 循环开始时添加空值检查
    while True:
        frame = image_capture_and_saver.get_latest()
        if frame is None or frame['rgb'] is None:  # 添加保护
            print("等待第一帧数据...")
            time.sleep(0.1)
        else:
            break
    print("******Begin save images!********")
    try:
        for step in range(episode_length-1):
            frame = image_capture_and_saver.get_latest()


            # 修复1：添加.copy()确保内存连续性
            rgb_img = frame['rgb'][:,:,::-1].copy()  # BGR->RGB
            # 更新观测数据
            observation["3rd"] = {
                'rgb': rgb_img,  # 转换后的RGB
            }

            current_gripper = gripper_open_flag(fa,gripper_thres) # 1 开 0关
            
            current_time = (1. - (step / float(episode_length - 1))) * 2. - 1.
            observation['low_dim_state'] = np.concatenate(
                [[current_gripper], [current_time]]).astype(np.float32)
            
            observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", frame['pcd']).copy()
            
            import open3d as o3d
            def vis_pcd(pcd, rgb):

                # 将点云和颜色转换为二维的形状 (N, 3)
                pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
                rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

                # 将点云和颜色信息保存为 PLY 文件
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
                pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
                # o3d.io.write_point_cloud(save_path, pcd)
                o3d.visualization.draw_geometries([pcd])
            # vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"]) #TODO
                    
            for key, v in observation.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if sub_k in ["rgb", "pcd"]:
                            # 确保数组内存连续
                            arr = np.transpose(v[sub_k], [2, 0, 1])
                            if not arr.flags.contiguous:
                                arr = np.ascontiguousarray(arr)
                            v[sub_k] = torch.from_numpy(arr).to(device).unsqueeze(0).float().contiguous()
                elif isinstance(v, np.ndarray):
                    if not v.flags.contiguous:
                        v = np.ascontiguousarray(v)
                    observation[key] = torch.from_numpy(v).to(device).unsqueeze(0).contiguous()


            target_pos, target_quat_, target_gripper = agent.act_real(observation,cameras_view)
            target_quat=[target_quat_[3],target_quat_[0],target_quat_[1],target_quat_[2]]
            # target_quat=target_quat_
            if target_quat[0] <0 :
                print("quat changed!")
                print("before quat:",target_quat)
                target_quat=np.array(target_quat)
                target_quat=target_quat*(-1)
                print("after quat:",target_quat)
            x_range = (0.15, 0.85)
            y_range = (-0.4, 0.6)
            z_range = (-0.05, 0.60)
            
            target_pos[0] = np.clip(target_pos[0], x_range[0], x_range[1])
            target_pos[1] = np.clip(target_pos[1], y_range[0], y_range[1])
            target_pos[2] = np.clip(target_pos[2], z_range[0], z_range[1])
            
            if target_pos[-1] < 0.01:
                target_pos[-1] = 0.015  #
                
            print("Predicted target pos: ", target_pos, "Predicted target quat: ", target_quat, "Predicted target gripper: ", target_gripper)
            osc_move_eval(fa,(target_pos, target_quat, np.array([target_gripper])),gripper_thres=gripper_thres,recording=False, recording_continous_frames=False, cameras=None)
    
    except Exception as e:
        import traceback
        print(f"主程序发生错误: {str(e)}")
        traceback.print_exc()
        # 自动停止图像保存线程
        image_capture_and_saver.stop()
        raise  # 可以选择不重新抛出异常
    finally:
        # 确保线程停止
        image_capture_and_saver.stop()
        print("已安全停止图像保存线程")


if __name__ == "__main__":
    _eval()