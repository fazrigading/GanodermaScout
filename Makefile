.PHONY: help up up-s3 down logs restart clean test

help:
	@echo "GanodermaScout CLI targets:"
	@echo "  make up          - Start core infrastructure (postgres, redis). Image files use local disk volume."
	@echo "  make up-s3       - Start core infrastructure plus SeaweedFS S3-compatible store (s3 profile)."
	@echo "  make down        - Stop all containers"
	@echo "  make logs        - Tail container logs"
	@echo "  make restart     - Restart all containers"
	@echo "  make clean       - Remove containers and temporary cache files"

up:
	docker compose up -d postgres redis

up-s3:
	docker compose --profile s3 up -d postgres redis seaweedfs

down:
	docker compose down

logs:
	docker compose logs -f

restart: down up

clean:
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
