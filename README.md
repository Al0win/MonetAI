Hey!

This is the project I created for my Generative AI Course.


Cloning:
git clone https://github.com/Al0win/MonetAI.git
cd MonetAI



Docker:
docker build -t my-jupyter-cuda-cudnn .
docker run --rm -it --gpus all -p 8888:8888 -v $(pwd)/monetai:/workspace my-jupyter-cuda-cudnn

## Development Workflow

1. Before starting new work:
```bash
git pull origin main  # Get latest changes
```

2. After making changes:
```bash
git add .
git commit -m "Description of changes"
git push origin main  # Or your feature branch
```


