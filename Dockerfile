FROM eidos-service.di.unito.it/eidos-base-pytorch:2.2.1

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

RUN pip3 install einops
RUN pip3 install plotly
RUN pip3 install plotly-utils
RUN pip3 install jaxtyping
# RUN pip3 install huggingface_hub
# RUN pip3 install openai
# RUN pip3 install sae_lens 
RUN pip3 install tabulate

# RUN pip3 install datasets==2.21.0 

ENTRYPOINT ["python3"]
