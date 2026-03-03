import argparse

def get_args():
    parser = argparse.ArgumentParser()

    ## 작물 선택
    parser.add_argument('--plant', type=str, default='butterhead', help='Type of plant ex) butterhead')                  
    parser.add_argument('--order1', type=str, default='2', help='몇 차 재배')  
    parser.add_argument('--order2_test', type=int, default=1, help='Which order2 value to use for testing (1, 2, or 3)')  


    ## 테스크 선택
    parser.add_argument('--Y_column', type=str, default='weight_last', choices=['day', 'weight_day', 'weight_last'],  
                            help="예측 테스크: 생육일자(day), 현재중량(weight_day), 최종중량(weight_last)")
    ## 증강 배수 선택
    parser.add_argument('--augmentation_type', default=0, type=int, help='몇배증강')  

    ## 실험 세팅
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU index to use (default: 0)')  # 사용할 GPU 번호 설정
    parser.add_argument('--model_name', type=str, default='resnet18', 
                            choices=['resnet18', 'vgg16_bn', 'efficientnet_b2', 'densenet121'],
                            help='학습할 모델의 이름 (resnet18, vgg16_bn, efficientnet_b2, densenet121)')    
    
    parser.add_argument('--model_name', type=str, default='efficientnet_b2', 
                    choices=['resnet18', 'vgg16_bn', 'efficientnet_b2', 'efficientnet_v2_s', 'vit_b_16', 'densenet121'],
                    help='학습할 모델의 이름')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005, help='학습률 (learning rate)')
    parser.add_argument('--step_size', type=int, default=70, help='LR 스케줄러의 step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR 스케줄러의 감마 값')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()