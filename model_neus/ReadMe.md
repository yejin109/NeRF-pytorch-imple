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
        - color fine : NeRF를 사용할 경우 이를 고려해서 render_core에서 color 계산
        - s_val : color 계산 시 사용한 값으로 [model]deviation network가 계산
        - cdf_fine: Sigmoid(SDF)로 계산한 값
        - gradient_error : SDF network에서 normal vector 계산한 것
        - weight : volumetric rendering으로 계산할 때 alpha 값으로 계산한 weight 값들

Process
1. Preprocess
    - z_vals 생성 
        - [config]n_samples만큼 sampling
        - (0, 1)의 범위로 한정한 다음 (near, far)로 
        - 앞으로 생각할 때 z_val의 값이 결국 paper에서 t값의 sequence로 보면 된다.
    - z_val 업데이트
        - ray의 위치를 나타내기 위해서 (-1/[config]n_samples, 1/[config]n_samples)의 uniform 분포에서 sampling한 것을 z_vals에 추가
        - 이는 sampling design에서 그대로 사용하는 것이 아니라 약간의 perturbation을 주는 것으로 ㅗ보인다. 
        - [config]perturb는 말 그대로 z값을 sampling할 때 perturbation을 줄지 말지 결정
2.(NeRF 사용시) Preprocess
    - (NeRF 사용시) z_vals_outside 생성
        - [config]n_outside (1e-3,~~)의 범위로 추가적으로 점을 더 뽑게 됨.
    - (NeRF 사용시) z_vals_outside 업데이트
        - far 기준으로 값을 업데이트하게 되는데 이 효과는 확실치 않음
3. Up sampling loop
    - overall
        - [config]up_sample_steps로 횟수 지정
        - [config]n_importance를 지정한 경우 
        - [model]SDF network를 여기서 이용하지만 gradient 계산은 하지 않음. FF loop에 포함되지 않고 inference로 이용하는 듯. rays_o와 rays_v를 가지고 SDF 계산          
        - 이를 바탕으로 z_vals 업데이트. 총 sample 수는 [config]n_samples와 [config]n_importance의 수
    - Up sampling loop : up_sample implementation
        - 계산한 SDF를 이용해서 해당하는 vector를 따라 기을기를 계산
        - (-1e3, 0)으로 clamp를 진행해서 해당 범위 외 값은 min, max로 대체
        - alpha값은 paper Equation 13, 22에서 언급한 대로 계산을 진행함.
        - inverse s 값은 여기서 결과적으로 iteration을 통해서 업데이트를 하는데 이 과정은 사실 Feed forward에 무관하게 하기에 [model]deviation network를 이용하지 않는 것으로 보인다.
        - 과정 : robust한 SDF 값을 구함 -> weight를 discrete opacity(alpha)로 계산 -> 이를 바탕으로 Z-sample을 계산
        - 결국 opacity에 따라서 sampling을 하게 되고, 즉 SDF=0인 지점을 중심으로 연산을 하게 된다!
    - Up sampling loop : sample_pdf Implementation
        - Up sampling 고과정에서 weight를 바탕으로 sampling 진행
        - 구현은 original NeRF 소스를 사용
    - Up sampling loop : cat_z_vals, SDF 업데이트 Z_vals sorting
        - 앞서 기존에 가지고 있던 z_vals와 함께 up_sample에서 만든 new_z_vals를 가지고 새로 SDF를 계산하게 된다.
        - 그렇게 계산한 SDF를 update업데이트를 하고, z_value는 sorting 진행
        - 마지막 Up sampling loop iteration이라면 z value sorting만 진행.
        - 결과적으로 opacity가 높은 곳에서 더 sampling을 하게 된다.
4. (NeRF 사용시) Background model
    - overall
        - background의 값을 계산
    - render_core_outside implementation
        - [model]NeRF network로 density와 sampled color 계산
        - 다만 이 scope에서 얘기하는 alpha는 NeRF에서 얘기하는 alpha
5. Render Core
    - overall
        - SDF 계산 -> IDR 기반 color 계산 & [model]Derivation Network 로 inverse s 계산 -> Taylor linearization 기반 SDF 업데이트
    - IDR 기반 color 계산
        - [model]Color Network를 이용해서 IDR 기반으로 color(Radiance 계산). 
        - 이 때 normal은 SDF의 gradient로 계산
        - Global light effect에 대한 정보는 결과적으로 SDF network의 정보를 이용한다는 것
    - (NeRF 사용시) background alpha를 계산하게 되었는데, 이게 결국 inside sphere 즉 ray의 point의 위치가 벗어난 곳에서의 color를 NeRF로 계산하고 ㄱ함께 사용한다는 것
    

## Inference

앞서 구현한 Train loop에서 gen_random_rays_at() 대신에 gen_rays_at()를 사용해서 이미지에서 값을 가져오도록 한다.
- random at는 전체 이미지에서 randomly 점들을 잡아서 계산을 하게 했다면, 여기에선 linspace로 모든 점을 사용하도록 한다.