FROM almalinux:9

# Install required dependencies
RUN dnf install -y \
    tbb \
    geos \
    libstdc++ \
    && dnf clean all

# Create working directory
WORKDIR /mrsid

# Copy the MrSID SDK
COPY MrSID_DSDK-9.5.5.5244-rhel9.x86-64.gcc1131/Raster_DSDK /mrsid/Raster_DSDK

# Set library path
ENV LD_LIBRARY_PATH=/mrsid/Raster_DSDK/bin:$LD_LIBRARY_PATH
ENV PATH=/mrsid/Raster_DSDK/bin:$PATH

# Create directory for input/output files
RUN mkdir -p /data

WORKDIR /data

# Default command shows help
CMD ["mrsiddecode"]