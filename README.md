# References
1. pytorch repo : [link](https://github.com/yenchenlin/nerf-pytorch/tree/1f064835d2cca26e4df2d7d130daa39a8cee1795)
2. volumetric rendering in NeRF [link](https://keras.io/examples/vision/nerf/)
3. Camera calibration [link](https://www.mathworks.com/help/vision/ug/camera-calibration.html)


# Log
## Performance, Error 
- [230420, ray] : ray.get_rays_np()에서 기존 구현에서 문제가 있었던 것 수정
- [230420, model] : Embedding에서 multries에 따라서 model 구조가 달라지고 있었음

## Profiling
- [230419, ray generation] : 현재 foward pass는 3.3it/s로 최적화 함
- [230419, render image] : redering.render_image()에서 render_rays가 2초 * chunk 3개 ~ 7s정도가 나오는 것 확인. 이는 ray가 H*W개를 만들어서 하다보니 render_rays()에서 ray 개수에 linear하게 시간이 증가한 것으로 봄
    - Render_rays 최적화 하기
- [230419, render image] : render pose 당 21s로 확인됨. 
- [230419, OOM] : Forward pass에서 내부 연산이후 아직 세션이 연결되어 있다보니 torch cache를 비우지 않으면 OOM이 나오는 상황.
    - back prop 이후에 cache 비우기
- [230419, OOM] Run network에서 1.6G정도 사용
- [230419, pose matrix]: LLFF dataset에서 Pose matrix를 처리할 때 transpose하여 진행하는 것이 존재. (LLFF.load_matrices()참고) 
    - 이렇게굳이 하는 이유를 파악해서 정리하기


## Parameter effect
- [230420, shuffling]: ray.prepare_ray_batching()에서 np.random.shuffle(rays_rgb)을 하지 않게 되면 이상하게 나옴


## Bottleneck
- ray generation, batch size dependent
- render ray, img size dependent

