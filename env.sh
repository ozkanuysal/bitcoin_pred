alias ui="uv run streamlit run main.py"
alias test="uv run test.py"

export APP_NAME="btc-forecast"
export IMAGE_NAME="$APP_NAME-image"
export CONTAINER_NAME="$APP_NAME-container"
export DOCKER_DEFAULT_PLATFORM=linux/amd64

alias build="docker build -t $IMAGE_NAME ."
alias run="echo "http://localhost:8501" && docker run --name $CONTAINER_NAME -p 8501:8501 -v $(pwd)/src:/app/src $IMAGE_NAME"
alias start="echo "http://localhost:8501" && docker start -a $CONTAINER_NAME"
alias stop="docker stop $CONTAINER_NAME"
alias reload="docker restart $CONTAINER_NAME"
alias launch="fly launch --name $APP_NAME --no-deploy"
alias deploy="fly deploy"