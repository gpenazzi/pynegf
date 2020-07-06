FROM quay.io/pypa/manylinux2014_x86_64

ENV PLAT manylinux2014_x86_64

# cmake > 3.5 needed.
RUN /opt/python/cp37-cp37m/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake
RUN ln -fs /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake

# System python3 must be present to execute the scripts building API wrappers.
RUN yum install -y python3 openblas-devel.x86_64 openblas.x86_64

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD tail -f /dev/null
