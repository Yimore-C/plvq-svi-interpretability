from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import os

def run_segmentation(input_dir, output_dir):
    # (需要先安装 mmseg)
    config_file = 'configs/segnext/segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512.py'
    checkpoint_file = 'segnext_mscan-l_1x16_512x512_adamw_160k_ade20k_20230209_172055-19b14b63.pth'
    
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_list = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]
    for img_name in img_list:
        img_path = os.path.join(input_dir, img_name)
        result = inference_model(model, img_path)
        
        out_file = os.path.join(output_dir, img_name)
        model.show_result(img_path, result, out_file=out_file, opacity=0.6)
        print(f"Processed: {img_name}")

if __name__ == '__main__':
    run_segmentation('data/sample_images', 'outputs/segmentation_results')
