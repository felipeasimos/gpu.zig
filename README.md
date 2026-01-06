## Roadmap

- [ ] modularize GPU operations (with tests for each part)
   - [ ] pipeline builder
      - [x] compute pipeline
         - [x] pipeline layout
            - [x] descriptor set layouts
               - [x] bindings
                   - [x] buffer
            - [x] push constantss
         - [x] shaders
      - [ ] graphics pipeline
- [ ] Review SVTree serialization

## Build and Run

```
apt install libvulkan1 libvulkan-dev vulkan-validationlayers glslc
```
