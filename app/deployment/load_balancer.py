"""Load balancer configuration and integration."""
from __future__ import annotations

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import uuid


class LoadBalancerType(Enum):
    """Load balancer types."""
    NGINX = "nginx"
    HAPROXY = "haproxy"
    TRAEFIK = "traefik"
    AWS_ALB = "aws_alb"
    KUBERNETES = "kubernetes"


class HealthCheckType(Enum):
    """Health check types."""
    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint."""
    endpoint_id: str
    host: str
    port: int
    weight: int = 100
    enabled: bool = True
    health_check_path: str = "/api/health"
    metadata: Dict[str, Any] = None


@dataclass
class HealthCheck:
    """Health check configuration."""
    check_type: HealthCheckType
    path: str = "/api/health"
    interval_seconds: int = 30
    timeout_seconds: int = 10
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    expected_status_codes: List[int] = None


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    config_id: str
    name: str
    lb_type: LoadBalancerType
    listen_port: int
    endpoints: List[ServiceEndpoint]
    health_check: HealthCheck
    
    # Load balancing algorithms
    algorithm: str = "round_robin"  # round_robin, least_conn, ip_hash, weighted
    
    # SSL/TLS configuration
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Rate limiting
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 1000
    
    # Sticky sessions
    sticky_sessions_enabled: bool = False
    sticky_sessions_cookie_name: str = "SESSIONID"
    
    # Timeouts
    connect_timeout_seconds: int = 5
    request_timeout_seconds: int = 30
    
    metadata: Dict[str, Any] = None


class ConfigGenerator:
    """Generates load balancer configuration files."""
    
    def __init__(self):
        self.generators = {
            LoadBalancerType.NGINX: self._generate_nginx_config,
            LoadBalancerType.HAPROXY: self._generate_haproxy_config,
            LoadBalancerType.TRAEFIK: self._generate_traefik_config,
            LoadBalancerType.AWS_ALB: self._generate_aws_alb_config,
            LoadBalancerType.KUBERNETES: self._generate_k8s_config
        }
    
    def generate_config(self, config: LoadBalancerConfig) -> str:
        """Generate configuration file for the specified load balancer type."""
        generator = self.generators.get(config.lb_type)
        if not generator:
            raise ValueError(f"Unsupported load balancer type: {config.lb_type}")
        
        return generator(config)
    
    def _generate_nginx_config(self, config: LoadBalancerConfig) -> str:
        """Generate NGINX configuration."""
        
        # Upstream configuration
        upstream_name = f"backend_{config.config_id}"
        upstream_config = f"upstream {upstream_name} {{\\n"
        
        # Load balancing method
        if config.algorithm == "least_conn":
            upstream_config += "    least_conn;\\n"
        elif config.algorithm == "ip_hash":
            upstream_config += "    ip_hash;\\n"
        
        # Backend servers
        for endpoint in config.endpoints:
            if endpoint.enabled:
                weight_clause = f" weight={endpoint.weight}" if config.algorithm == "weighted" else ""
                upstream_config += f"    server {endpoint.host}:{endpoint.port}{weight_clause};\\n"
        
        upstream_config += "}\\n\\n"
        
        # Server configuration
        server_config = f"""server {{
    listen {config.listen_port};
    server_name localhost;
    
    # Timeouts
    proxy_connect_timeout {config.connect_timeout_seconds}s;
    proxy_send_timeout {config.request_timeout_seconds}s;
    proxy_read_timeout {config.request_timeout_seconds}s;
    
    # Health check location
    location {config.health_check.path} {{
        access_log off;
        proxy_pass http://{upstream_name};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }}
    
    # Main application
    location / {{
        proxy_pass http://{upstream_name};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Enable request buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # Enable response compression
        gzip on;
        gzip_types text/plain application/json application/xml text/css text/javascript;
        """
        
        # Rate limiting
        if config.rate_limit_enabled:
            limit_zone = f"limit_zone_{config.config_id}"
            server_config = f"""# Rate limiting
limit_req_zone $binary_remote_addr zone={limit_zone}:10m rate={config.rate_limit_requests_per_minute}r/m;

""" + server_config
            server_config += f"        limit_req zone={limit_zone} burst=5 nodelay;\\n"
        
        # Sticky sessions
        if config.sticky_sessions_enabled:
            server_config += f"""        
        # Sticky sessions
        proxy_cookie_path / "/; HttpOnly; Secure";
        """
        
        server_config += "    }\\n"
        
        # SSL configuration
        if config.ssl_enabled and config.ssl_cert_path and config.ssl_key_path:
            server_config = server_config.replace(f"listen {config.listen_port};", 
                                                 f"listen {config.listen_port} ssl;")
            server_config += f"""    
    ssl_certificate {config.ssl_cert_path};
    ssl_certificate_key {config.ssl_key_path};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    """
        
        server_config += "}\\n"
        
        return upstream_config + server_config
    
    def _generate_haproxy_config(self, config: LoadBalancerConfig) -> str:
        """Generate HAProxy configuration."""
        
        haproxy_config = f"""global
    daemon
    log stdout local0
    maxconn 4096

defaults
    mode http
    timeout connect {config.connect_timeout_seconds}s
    timeout client {config.request_timeout_seconds}s
    timeout server {config.request_timeout_seconds}s
    option httplog
    option dontlognull
    option http-server-close
    option redispatch
    retries 3

frontend web_frontend
    bind *:{config.listen_port}"""
        
        if config.ssl_enabled and config.ssl_cert_path:
            haproxy_config += f" ssl crt {config.ssl_cert_path}"
        
        haproxy_config += f"""
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    """
        
        if config.rate_limit_enabled:
            requests_per_10s = config.rate_limit_requests_per_minute // 6
            haproxy_config += f"    http-request deny if {{ sc_http_req_rate(0) gt {requests_per_10s} }}\\n"
        
        haproxy_config += f"""    
    # Health check
    acl is_health_check path {config.health_check.path}
    use_backend web_backend if is_health_check
    
    default_backend web_backend

backend web_backend
    """
        
        # Load balancing algorithm
        if config.algorithm == "round_robin":
            haproxy_config += "    balance roundrobin\\n"
        elif config.algorithm == "least_conn":
            haproxy_config += "    balance leastconn\\n"
        elif config.algorithm == "ip_hash":
            haproxy_config += "    balance source\\n"
        
        # Health check configuration
        health_check_config = f"check inter {config.health_check.interval_seconds}s"
        if config.health_check.check_type == HealthCheckType.HTTP:
            health_check_config += f" httpchk GET {config.health_check.path}"
        
        # Backend servers
        for i, endpoint in enumerate(config.endpoints):
            if endpoint.enabled:
                weight_clause = f" weight {endpoint.weight}" if config.algorithm == "weighted" else ""
                haproxy_config += f"    server backend{i+1} {endpoint.host}:{endpoint.port} {health_check_config}{weight_clause}\\n"
        
        # Sticky sessions
        if config.sticky_sessions_enabled:
            haproxy_config += f"""
    # Sticky sessions
    cookie {config.sticky_sessions_cookie_name} insert indirect nocache
    """
            # Add cookie to each server
            lines = haproxy_config.split('\\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('server backend'):
                    lines[i] += f" cookie backend{i+1}"
            haproxy_config = '\\n'.join(lines)
        
        return haproxy_config
    
    def _generate_traefik_config(self, config: LoadBalancerConfig) -> str:
        """Generate Traefik configuration (YAML format)."""
        
        # Service configuration
        service_name = f"rag-service-{config.config_id}"
        
        traefik_config = {
            "http": {
                "routers": {
                    f"{service_name}-router": {
                        "rule": f"Host(`localhost`) && PathPrefix(`/`)",
                        "service": service_name,
                        "entryPoints": ["web"]
                    }
                },
                "services": {
                    service_name: {
                        "loadBalancer": {
                            "servers": [],
                            "healthCheck": {
                                "path": config.health_check.path,
                                "interval": f"{config.health_check.interval_seconds}s",
                                "timeout": f"{config.health_check.timeout_seconds}s"
                            }
                        }
                    }
                }
            },
            "entryPoints": {
                "web": {
                    "address": f":{config.listen_port}"
                }
            }
        }
        
        # Add backend servers
        for endpoint in config.endpoints:
            if endpoint.enabled:
                server_config = {"url": f"http://{endpoint.host}:{endpoint.port}"}
                if config.algorithm == "weighted":
                    server_config["weight"] = endpoint.weight
                
                traefik_config["http"]["services"][service_name]["loadBalancer"]["servers"].append(server_config)
        
        # Load balancing algorithm
        if config.algorithm == "least_conn":
            traefik_config["http"]["services"][service_name]["loadBalancer"]["responseForwarding"] = {
                "flushInterval": "1ms"
            }
        
        # SSL configuration
        if config.ssl_enabled:
            traefik_config["entryPoints"]["websecure"] = {
                "address": f":{config.listen_port}"
            }
            traefik_config["http"]["routers"][f"{service_name}-router"]["entryPoints"] = ["websecure"]
            if config.ssl_cert_path:
                traefik_config["http"]["routers"][f"{service_name}-router"]["tls"] = {
                    "certFile": config.ssl_cert_path,
                    "keyFile": config.ssl_key_path
                }
        
        # Rate limiting
        if config.rate_limit_enabled:
            traefik_config["http"]["middlewares"] = {
                "rate-limit": {
                    "rateLimit": {
                        "burst": 10,
                        "period": "1m",
                        "average": config.rate_limit_requests_per_minute
                    }
                }
            }
            traefik_config["http"]["routers"][f"{service_name}-router"]["middlewares"] = ["rate-limit"]
        
        # Convert to YAML-like string
        import yaml
        return yaml.dump(traefik_config, default_flow_style=False)
    
    def _generate_aws_alb_config(self, config: LoadBalancerConfig) -> str:
        """Generate AWS Application Load Balancer configuration (Terraform format)."""
        
        alb_name = f"rag-alb-{config.config_id}"
        target_group_name = f"rag-tg-{config.config_id}"
        
        terraform_config = f'''
# Application Load Balancer
resource "aws_lb" "{alb_name}" {{
  name               = "{alb_name}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = var.public_subnet_ids
  
  enable_deletion_protection = false
  
  tags = {{
    Name = "RAG System ALB"
    Environment = var.environment
  }}
}}

# Target Group
resource "aws_lb_target_group" "{target_group_name}" {{
  name     = "{target_group_name}"
  port     = {config.listen_port}
  protocol = "HTTP"
  vpc_id   = var.vpc_id
  
  health_check {{
    enabled             = true
    healthy_threshold   = {config.health_check.healthy_threshold}
    unhealthy_threshold = {config.health_check.unhealthy_threshold}
    timeout             = {config.health_check.timeout_seconds}
    interval            = {config.health_check.interval_seconds}
    path                = "{config.health_check.path}"
    matcher             = "200"
    port                = "traffic-port"
    protocol            = "HTTP"
  }}
  
  # Load balancing algorithm
  load_balancing_algorithm_type = "round_robin"
  
  # Sticky sessions
  '''
        
        if config.sticky_sessions_enabled:
            terraform_config += f'''
  stickiness {{
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }}
  '''
        
        terraform_config += '''
  
  tags = {
    Name = "RAG System Target Group"
  }
}

# Target Group Attachments
'''
        
        for i, endpoint in enumerate(config.endpoints):
            if endpoint.enabled:
                terraform_config += f'''
resource "aws_lb_target_group_attachment" "tg_attachment_{i}" {{
  target_group_arn = aws_lb_target_group.{target_group_name}.arn
  target_id        = "{endpoint.host}"  # Instance ID or IP
  port             = {endpoint.port}
}}
'''
        
        # Listener configuration
        terraform_config += f'''
# HTTP Listener
resource "aws_lb_listener" "web" {{
  load_balancer_arn = aws_lb.{alb_name}.arn
  port              = "{config.listen_port}"
  protocol          = "HTTP"
  
  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.{target_group_name}.arn
  }}
}}
'''
        
        # SSL listener if enabled
        if config.ssl_enabled:
            terraform_config += f'''
# HTTPS Listener
resource "aws_lb_listener" "web_ssl" {{
  load_balancer_arn = aws_lb.{alb_name}.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.ssl_certificate_arn
  
  default_action {{
    type             = "forward"
    target_group_arn = aws_lb_target_group.{target_group_name}.arn
  }}
}}
'''
        
        # Security group
        terraform_config += '''
# Security Group for ALB
resource "aws_security_group" "alb_sg" {
  name_prefix = "rag-alb-sg"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "RAG ALB Security Group"
  }
}
'''
        
        return terraform_config
    
    def _generate_k8s_config(self, config: LoadBalancerConfig) -> str:
        """Generate Kubernetes configuration (YAML format)."""
        
        service_name = f"rag-service-{config.config_id}"
        deployment_name = f"rag-deployment-{config.config_id}"
        
        k8s_config = f'''apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  labels:
    app: rag-system
spec:
  type: LoadBalancer
  ports:
  - port: {config.listen_port}
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: rag-system

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  labels:
    app: rag-system
spec:
  replicas: {len([e for e in config.endpoints if e.enabled])}
  selector:
    matchLabels:
      app: rag-system
  template:
    metadata:
      labels:
        app: rag-system
    spec:
      containers:
      - name: rag-api
        image: rag-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        livenessProbe:
          httpGet:
            path: {config.health_check.path}
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: {config.health_check.interval_seconds}
          timeoutSeconds: {config.health_check.timeout_seconds}
          failureThreshold: {config.health_check.unhealthy_threshold}
        readinessProbe:
          httpGet:
            path: {config.health_check.path}
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  nginx.conf: |
    upstream backend {{
'''
        
        # Add backend servers
        for endpoint in config.endpoints:
            if endpoint.enabled:
                k8s_config += f"      server {endpoint.host}:{endpoint.port};\\n"
        
        k8s_config += f'''    }}
    
    server {{
      listen {config.listen_port};
      location / {{
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      }}
      
      location {config.health_check.path} {{
        proxy_pass http://backend;
        access_log off;
      }}
    }}
'''
        
        if config.ssl_enabled:
            k8s_config += '''
---
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: # base64 encoded certificate
  tls.key: # base64 encoded private key

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: tls-secret
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ''' + service_name + '''
            port:
              number: ''' + str(config.listen_port)
        
        return k8s_config


class LoadBalancerManager:
    """Manages load balancer configurations and deployment."""
    
    def __init__(self, storage_path: str = "./data/load_balancer"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.configs_file = self.storage_path / "lb_configs.json"
        self.configs_dir = self.storage_path / "generated_configs"
        self.configs_dir.mkdir(exist_ok=True)
        
        self.configs: Dict[str, LoadBalancerConfig] = {}
        self.config_generator = ConfigGenerator()
        
        # Load existing configurations
        self.load_configurations()
    
    def load_configurations(self):
        """Load load balancer configurations from storage."""
        if self.configs_file.exists():
            try:
                with open(self.configs_file, 'r') as f:
                    data = json.load(f)
                    
                    for config_data in data:
                        config_data['lb_type'] = LoadBalancerType(config_data['lb_type'])
                        config_data['health_check']['check_type'] = HealthCheckType(config_data['health_check']['check_type'])
                        
                        config_data['endpoints'] = [ServiceEndpoint(**ep) for ep in config_data['endpoints']]
                        config_data['health_check'] = HealthCheck(**config_data['health_check'])
                        
                        config = LoadBalancerConfig(**config_data)
                        self.configs[config.config_id] = config
                        
            except Exception as e:
                print(f"Failed to load load balancer configurations: {e}")
    
    def save_configurations(self):
        """Save load balancer configurations to storage."""
        try:
            configs_data = []
            for config in self.configs.values():
                data = asdict(config)
                data['lb_type'] = config.lb_type.value
                data['health_check']['check_type'] = config.health_check.check_type.value
                configs_data.append(data)
            
            with open(self.configs_file, 'w') as f:
                json.dump(configs_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save load balancer configurations: {e}")
    
    def create_load_balancer_config(self, name: str, lb_type: LoadBalancerType,
                                  listen_port: int, endpoints: List[ServiceEndpoint],
                                  **kwargs) -> LoadBalancerConfig:
        """Create a new load balancer configuration."""
        
        config_id = f"lb_{int(time.time())}"
        
        # Default health check
        health_check = kwargs.get('health_check', HealthCheck(
            check_type=HealthCheckType.HTTP,
            path="/api/health",
            interval_seconds=30,
            timeout_seconds=10,
            healthy_threshold=2,
            unhealthy_threshold=3,
            expected_status_codes=[200]
        ))
        
        config = LoadBalancerConfig(
            config_id=config_id,
            name=name,
            lb_type=lb_type,
            listen_port=listen_port,
            endpoints=endpoints,
            health_check=health_check,
            **{k: v for k, v in kwargs.items() if k != 'health_check'}
        )
        
        self.configs[config_id] = config
        self.save_configurations()
        
        return config
    
    def generate_configuration_file(self, config_id: str) -> Optional[str]:
        """Generate configuration file for a load balancer."""
        config = self.configs.get(config_id)
        if not config:
            return None
        
        try:
            config_content = self.config_generator.generate_config(config)
            
            # Determine file extension based on load balancer type
            extensions = {
                LoadBalancerType.NGINX: "conf",
                LoadBalancerType.HAPROXY: "cfg",
                LoadBalancerType.TRAEFIK: "yml",
                LoadBalancerType.AWS_ALB: "tf",
                LoadBalancerType.KUBERNETES: "yaml"
            }
            
            extension = extensions.get(config.lb_type, "conf")
            filename = f"{config.name}_{config_id}.{extension}"
            filepath = self.configs_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(config_content)
            
            print(f"✅ Generated {config.lb_type.value} configuration: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"Failed to generate configuration file: {e}")
            return None
    
    def update_endpoint_health(self, config_id: str, endpoint_id: str, 
                             is_healthy: bool) -> bool:
        """Update the health status of an endpoint."""
        config = self.configs.get(config_id)
        if not config:
            return False
        
        endpoint = next((ep for ep in config.endpoints if ep.endpoint_id == endpoint_id), None)
        if not endpoint:
            return False
        
        endpoint.enabled = is_healthy
        self.save_configurations()
        
        # Regenerate configuration if endpoint status changed
        self.generate_configuration_file(config_id)
        
        return True
    
    def add_endpoint(self, config_id: str, endpoint: ServiceEndpoint) -> bool:
        """Add a new endpoint to a load balancer configuration."""
        config = self.configs.get(config_id)
        if not config:
            return False
        
        # Check for duplicate endpoint IDs
        if any(ep.endpoint_id == endpoint.endpoint_id for ep in config.endpoints):
            return False
        
        config.endpoints.append(endpoint)
        self.save_configurations()
        self.generate_configuration_file(config_id)
        
        return True
    
    def remove_endpoint(self, config_id: str, endpoint_id: str) -> bool:
        """Remove an endpoint from a load balancer configuration."""
        config = self.configs.get(config_id)
        if not config:
            return False
        
        original_count = len(config.endpoints)
        config.endpoints = [ep for ep in config.endpoints if ep.endpoint_id != endpoint_id]
        
        if len(config.endpoints) < original_count:
            self.save_configurations()
            self.generate_configuration_file(config_id)
            return True
        
        return False
    
    def update_endpoint_weight(self, config_id: str, endpoint_id: str, weight: int) -> bool:
        """Update the weight of an endpoint for weighted load balancing."""
        config = self.configs.get(config_id)
        if not config:
            return False
        
        endpoint = next((ep for ep in config.endpoints if ep.endpoint_id == endpoint_id), None)
        if not endpoint:
            return False
        
        endpoint.weight = weight
        self.save_configurations()
        
        # Regenerate configuration if using weighted algorithm
        if config.algorithm == "weighted":
            self.generate_configuration_file(config_id)
        
        return True
    
    def get_load_balancer_status(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a load balancer configuration."""
        config = self.configs.get(config_id)
        if not config:
            return None
        
        total_endpoints = len(config.endpoints)
        healthy_endpoints = len([ep for ep in config.endpoints if ep.enabled])
        
        return {
            'config_id': config_id,
            'name': config.name,
            'lb_type': config.lb_type.value,
            'listen_port': config.listen_port,
            'algorithm': config.algorithm,
            'total_endpoints': total_endpoints,
            'healthy_endpoints': healthy_endpoints,
            'endpoint_health_percentage': (healthy_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0,
            'ssl_enabled': config.ssl_enabled,
            'rate_limit_enabled': config.rate_limit_enabled,
            'endpoints': [
                {
                    'endpoint_id': ep.endpoint_id,
                    'host': ep.host,
                    'port': ep.port,
                    'weight': ep.weight,
                    'enabled': ep.enabled
                }
                for ep in config.endpoints
            ]
        }
    
    def create_default_rag_config(self) -> LoadBalancerConfig:
        """Create a default load balancer configuration for the RAG system."""
        
        # Default endpoints (can be customized)
        endpoints = [
            ServiceEndpoint(
                endpoint_id="rag_api_1",
                host="localhost",
                port=8000,
                weight=100,
                enabled=True,
                health_check_path="/api/health"
            )
        ]
        
        # Create configuration for NGINX (most common)
        config = self.create_load_balancer_config(
            name="RAG System Load Balancer",
            lb_type=LoadBalancerType.NGINX,
            listen_port=80,
            endpoints=endpoints,
            algorithm="round_robin",
            ssl_enabled=False,
            rate_limit_enabled=True,
            rate_limit_requests_per_minute=1000,
            connect_timeout_seconds=5,
            request_timeout_seconds=30
        )
        
        # Generate the configuration file
        self.generate_configuration_file(config.config_id)
        
        return config
    
    def get_deployment_instructions(self, config_id: str) -> Optional[Dict[str, str]]:
        """Get deployment instructions for a load balancer configuration."""
        config = self.configs.get(config_id)
        if not config:
            return None
        
        config_file = self.generate_configuration_file(config_id)
        if not config_file:
            return None
        
        instructions = {
            LoadBalancerType.NGINX: f"""
# NGINX Deployment Instructions

1. Copy the configuration file to NGINX:
   sudo cp {config_file} /etc/nginx/sites-available/{config.name}
   sudo ln -s /etc/nginx/sites-available/{config.name} /etc/nginx/sites-enabled/

2. Test the configuration:
   sudo nginx -t

3. Reload NGINX:
   sudo systemctl reload nginx

4. Enable NGINX to start on boot:
   sudo systemctl enable nginx
""",
            LoadBalancerType.HAPROXY: f"""
# HAProxy Deployment Instructions

1. Copy the configuration file:
   sudo cp {config_file} /etc/haproxy/haproxy.cfg

2. Test the configuration:
   sudo haproxy -f /etc/haproxy/haproxy.cfg -c

3. Restart HAProxy:
   sudo systemctl restart haproxy

4. Enable HAProxy to start on boot:
   sudo systemctl enable haproxy
""",
            LoadBalancerType.TRAEFIK: f"""
# Traefik Deployment Instructions

1. Copy the configuration file:
   cp {config_file} traefik.yml

2. Run Traefik with Docker:
   docker run -d \\
     --name traefik \\
     -p {config.listen_port}:{config.listen_port} \\
     -v $(pwd)/traefik.yml:/etc/traefik/traefik.yml \\
     traefik:v2.10

3. Or run with Docker Compose:
   Add the configuration to your docker-compose.yml
""",
            LoadBalancerType.AWS_ALB: f"""
# AWS ALB Deployment Instructions

1. Initialize Terraform:
   terraform init

2. Copy the configuration:
   cp {config_file} main.tf

3. Set variables in terraform.tfvars:
   vpc_id = "vpc-xxxxxxxx"
   public_subnet_ids = ["subnet-xxxxxxxx", "subnet-yyyyyyyy"]
   environment = "production"

4. Plan and apply:
   terraform plan
   terraform apply
""",
            LoadBalancerType.KUBERNETES: f"""
# Kubernetes Deployment Instructions

1. Apply the configuration:
   kubectl apply -f {config_file}

2. Check the service status:
   kubectl get services
   kubectl get pods

3. Get the external IP:
   kubectl get service {config.name}

4. Check logs if needed:
   kubectl logs -l app=rag-system
"""
        }
        
        return {
            'instructions': instructions.get(config.lb_type, "No specific instructions available"),
            'config_file_path': config_file,
            'lb_type': config.lb_type.value
        }
    
    def get_all_configurations(self) -> List[Dict[str, Any]]:
        """Get summary of all load balancer configurations."""
        return [
            {
                'config_id': config.config_id,
                'name': config.name,
                'lb_type': config.lb_type.value,
                'listen_port': config.listen_port,
                'endpoint_count': len(config.endpoints),
                'healthy_endpoints': len([ep for ep in config.endpoints if ep.enabled]),
                'algorithm': config.algorithm,
                'ssl_enabled': config.ssl_enabled
            }
            for config in self.configs.values()
        ]
