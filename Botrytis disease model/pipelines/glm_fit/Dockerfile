ARG BASE_IMAGE

FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

RUN install2.r --error --skipinstalled renv

RUN apt update \ 
    && apt install libxt-dev curl jags -y \ 
    && rm -rf /var/lib/apt/lists/*

ARG QUARTO_VERSION

RUN mkdir -p /opt/quarto/${QUARTO_VERSION} \
    && curl -o quarto.tar.gz -L "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.tar.gz" \
    && tar -zxvf quarto.tar.gz -C "/opt/quarto/${QUARTO_VERSION}" --strip-components=1 \
    && rm quarto.tar.gz \
    && rm -rf /usr/local/bin/quarto \
    && ln -s "/opt/quarto/${QUARTO_VERSION}/bin/quarto" /usr/local/bin/quarto

RUN quarto install tinytex 