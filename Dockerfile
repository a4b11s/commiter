FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app/

COPY src/ .

ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV KERAS_BACKEND="tensorflow"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade keras-nlp \
    && pip install --upgrade keras \
    && pip install -U tensorboard_plugin_profile

ENTRYPOINT [ "python", "-u", "-m", "main"]