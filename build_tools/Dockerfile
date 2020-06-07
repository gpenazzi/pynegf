FROM quay.io/pypa/manylinux2014_x86_64

ENV PLAT manylinux2010_x86_64

COPY entrypoint.sh /entrypoint.sh
RUN /opt/python/cp37-cp37m/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake
RUN ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
