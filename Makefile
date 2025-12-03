.PHONY: help build shell test run examples jupyter clean coverage

help:
	@echo "Beagle Development Commands"
	@echo "============================"
	@echo ""
	@echo "Core Commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make shell      - Open bash shell in dev container"
	@echo "  make test       - Run all tests"
	@echo "  make coverage   - Run tests with coverage report"
	@echo "  make jupyter    - Start Jupyter Lab (fast interactive plotting)"
	@echo ""
	@echo "Advanced:"
	@echo "  make examples   - Open shell with example dependencies"
	@echo "  make run CMD=<command>  - Run arbitrary command in container"
	@echo "  make clean      - Remove containers and dangling images"
	@echo ""
	@echo "Environment Variables:"
	@echo "  MOUNT_DIR=/path/to/dir    - Mount external directory (default: /tmp/beagle-data)"
	@echo "  MOUNT_TARGET=/data        - Mount point in container (default: /data)"
	@echo "  HOST_PORT=8888            - Expose port on host (default: 8888)"
	@echo "  CONTAINER_PORT=8888       - Port in container (default: 8888)"
	@echo "  NVIDIA_VISIBLE_DEVICES=0,1 - Select specific GPUs (default: all GPUs)"
	@echo "  BEAGLE_HEADLESS=true      - Headless viz mode (default: true, use false for GUI)"
	@echo ""
	@echo "GPU Support:"
	@echo "  - Requires nvidia-docker2 or Docker with nvidia runtime"
	@echo "  - All GPUs are available by default in containers"
	@echo "  - Use NVIDIA_VISIBLE_DEVICES to limit GPU access"
	@echo "  - JAX will automatically use GPUs if available"
	@echo ""
	@echo "Examples:"
	@echo "  make shell"
	@echo "  MOUNT_DIR=~/data make shell"
	@echo "  HOST_PORT=9000 make shell"
	@echo "  NVIDIA_VISIBLE_DEVICES=0 make shell  # Use only GPU 0"
	@echo "  BEAGLE_HEADLESS=false make shell     # Enable GUI for visualization"
	@echo "  make run CMD='python -c \"import jax; print(jax.devices())\"'  # Verify GPU"
	@echo "  make run CMD='python examples/root_writer.py /data/input /data/output'"

build:
	docker-compose build

shell:
	docker-compose run --rm dev bash

test:
	docker-compose run --rm dev pytest tests/ -v

coverage:
	docker-compose run --rm dev pytest tests/ --cov=beagle --cov-report=term-missing

examples:
	docker-compose run --rm examples bash

jupyter:
	@echo "Starting Jupyter Lab..."
	@echo "Access at: http://localhost:8888"
	@echo "Press Ctrl+C to stop"
	docker-compose run --rm --service-ports dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

run:
	docker-compose run --rm dev $(CMD)

clean:
	docker-compose down
	docker system prune -f
