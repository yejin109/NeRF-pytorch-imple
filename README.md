# References
1. pytorch repo : [link](https://github.com/yenchenlin/nerf-pytorch/tree/1f064835d2cca26e4df2d7d130daa39a8cee1795)
2. volumetric rendering in NeRF [link](https://keras.io/examples/vision/nerf/)
3. Camera calibration [link](https://www.mathworks.com/help/vision/ug/camera-calibration.html)


# Log
## Profiling
- [230419, ray generation] : 현재 foward pass는 1.44s정도로 최적화 함. 이는 no-batching(use-batch)에 따라서 ray generation에서 10s까지 늘어나는 것을 줄인 것. 원래 코드에서도 사용하고 있지 않던 옵션 
    - 사용하지 않는 코드 블럭 줄이기
- [230419, render image] : redering.render_image()에서 render_rays가 2초 * chunk 3개 ~ 7s정도가 나오는 것 확인. 이는 ray가 H*W개를 만들어서 하다보니 render_rays()에서 ray 개수에 linear하게 시간이 증가한 것으로 봄
    - Render_rays 최적화 하기
- [230419, OOM] : Forward pass에서 내부 연산이후 아직 세션이 연결되어 있다보니 torch cache를 비우지 않으면 OOM이 나오는 상황.
    - back prop 이후에 cache 비우기

## Bottleneck
- ray generation, batch size dependent
- render ray, img size dependent

