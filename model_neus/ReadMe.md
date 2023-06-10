# Pipeline

## Train

### gen_random_rays_at
Purpose : Generate random rays at world space from one camera.

Input : img_idx, batch_size
- img_idx : torch.randperm으로 이미지 index sampling한 것
- batch_size : config에서 지정한 batch size

Output : 
- shape : (batch size, 10) 
    - 10 : rays_0, rays_v, color(rgb), mask(binary) 

Process
1. 전체 pixel 영역에서 임의로 x,y 위치를 선정(torch.randint)
2. 해당 위치의 color와 mask를 indexing
3. world coordinate에서의 ray의 위치 (intrinsic matrix)^{-1} @ [Index_x, Index_y, 1]
    - 마지막 -1은 homogeneous coordinate를 위해서 붙인 것.
    - 지금 idx가 결국 pixel의 위치를 말하니 intrinsic의 inverse와 곱하게 된다면 world coordinate에서의 좌표에 해당
4. normalization
    - 이 때 norm은 ord = 2, sum(abs(x)^{ord})^{(1 / ord)}으로 정의(L2 norm)
5. rays_v 
    - {world to coornidate, pose matrix} @ {(normalized) world coorniate에서 pixel의 위치}
    - camera coordinate에서의 pixel에서 origin으로의 방향을 나타내는 정보
6. rays_o
    - {pose matrix} 그대로 사용
    - camera coordinate에서의 pixel의 위치를 나타내는 정보


### near_far_from_sphere

Input : 
    - rays_o
    - rays_v

Output:
    - near
    - far

Process
1. a 계산 : sum(rays_d**2)
2. b 계산 : 2*sum(rays_0 * rays_d)
    - inner product인 듯
3. mid 계산 : 1/2 * (-b/a)
    - 왜 굳이 앞에서 2를 곱하고 2를 나누는지는 아마 numerical error 때문인 것 같다.
    - 그리고 rays_d는 normalize되어 있을 것
    - 결국 {- inner product}의 계산 목적. 이것은 위치 벡터와 방향벡터의 방향이 다르기 때문에 -를 곱하는 듯
4. near, far 계산
    - near = mid - 1.0
    - far = mid + 1.0

결과적으로 -1을 해서 near가 되는 것은 origin에서 멀어질수록 z값이 커지는 것과 동일하게 생각하면 될 것. 그 범위가 왜 +- 1인지는 확인해야 한다.


### render

Input
    - rays_o
    - rays_d
    - near
    - far
    - perturb_overwrite=-1
        - [config] perturb값을 지정해서 preprocess에서 사용
    - background_rgb=None :
    - cos_anneal_ratio=0.0
        - get_cos_anneal_ratio(iter_step, anneal_end)으로 계산
        - [config]anneal_end를 안쓰면 항상 1
        - 사용하게 된다면 min(1, iter_step/ anneal_end)가 되어서 end시점까지 train 시점을 알려주는 형태

Output
    - 예측 결과, Dict

Process
1. Preprocess
    - z_vals 생성 
        - [config]n_samples만큼 sampling
        - (0, 1)의 범위로 한정한 다음 (near, far)로 
    - (NeRF 사용시) z_vals_outside 생성
        - [config]n_outside (1e-3,~~)의 범위로 추가적으로 점을 더 뽑게 됨.
    - z_val 업데이트
        - ray의 위치를 나타내기 위해서 (-1/[config]n_samples, 1/[config]n_samples)의 uniform 분포에서 sampling한 것을 z_vals에 추가
        - 이는 sampling design에서 그대로 사용하는 것이 아니라 약간의 perturbation을 주는 것으로 ㅗ보인다. 
        - [config]perturb는 말 그대로 z값을 sampling할 때 perturbation을 줄지 말지 결정
    - (NeRF 사용시) z_vals_outside 업데이트
        - far 기준으로 값을 업데이트하게 되는데 이 효과는 확실치 않음
    - Up sampling
        - [config]n_importance를 지정한 경우 
        - rays_o와 rays_v를 가지고 SDF 계산
        - 이를 바탕으로 z_vals 업데이트
        - 총 sample 수는 [config]n_samples와 [config]n_importance의 합

## Inference 