"""User authentication and authorization system."""
from __future__ import annotations

import jwt
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext


class Role(Enum):
    """User roles with different permission levels."""
    VIEWER = "viewer"           # Can only view and query
    USER = "user"              # Can query and upload documents  
    MODERATOR = "moderator"    # Can manage documents and users
    ADMIN = "admin"            # Full system access


class Permission(Enum):
    """Granular permissions."""
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    QUERY_SYSTEM = "query_system"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_SYSTEM = "manage_system"
    ACCESS_ADMIN = "access_admin"


@dataclass
class User:
    """User model with authentication and authorization data."""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: Role
    permissions: List[Permission]
    document_access: List[str]  # Document IDs user can access
    created_at: float
    last_login: Optional[float]
    is_active: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class Session:
    """User session data."""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuthenticationError(Exception):
    """Authentication related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization related errors."""
    pass


class UserManager:
    """Manages user accounts, authentication, and sessions."""
    
    def __init__(self, storage_path: str = "./data/auth"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.users_file = self.storage_path / "users.json"
        self.sessions_file = self.storage_path / "sessions.json"
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT settings
        self.jwt_secret = self._get_or_create_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = 30
        
        # Storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Role permissions mapping
        self.role_permissions = {
            Role.VIEWER: [Permission.READ_DOCUMENTS, Permission.QUERY_SYSTEM],
            Role.USER: [
                Permission.READ_DOCUMENTS, Permission.WRITE_DOCUMENTS,
                Permission.QUERY_SYSTEM
            ],
            Role.MODERATOR: [
                Permission.READ_DOCUMENTS, Permission.WRITE_DOCUMENTS,
                Permission.DELETE_DOCUMENTS, Permission.QUERY_SYSTEM,
                Permission.VIEW_ANALYTICS
            ],
            Role.ADMIN: list(Permission)  # All permissions
        }
        
        # Load existing data
        self.load_users()
        self.load_sessions()
        
        # Create default admin if no users exist
        if not self.users:
            self.create_default_admin()
    
    def _get_or_create_jwt_secret(self) -> str:
        """Get or create JWT secret key."""
        secret_file = self.storage_path / "jwt_secret.txt"
        
        if secret_file.exists():
            return secret_file.read_text().strip()
        else:
            secret = secrets.token_urlsafe(32)
            secret_file.write_text(secret)
            return secret
    
    def load_users(self):
        """Load users from storage."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data:
                        user_data['role'] = Role(user_data['role'])
                        user_data['permissions'] = [Permission(p) for p in user_data['permissions']]
                        user = User(**user_data)
                        self.users[user.user_id] = user
            except Exception as e:
                print(f"Failed to load users: {e}")
    
    def save_users(self):
        """Save users to storage."""
        try:
            users_data = []
            for user in self.users.values():
                data = asdict(user)
                data['role'] = user.role.value
                data['permissions'] = [p.value for p in user.permissions]
                users_data.append(data)
            
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save users: {e}")
    
    def load_sessions(self):
        """Load sessions from storage."""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    for session_data in data:
                        session = Session(**session_data)
                        # Only keep non-expired sessions
                        if session.expires_at > time.time():
                            self.sessions[session.session_id] = session
            except Exception as e:
                print(f"Failed to load sessions: {e}")
    
    def save_sessions(self):
        """Save sessions to storage."""
        try:
            # Clean expired sessions before saving
            current_time = time.time()
            active_sessions = {
                sid: session for sid, session in self.sessions.items()
                if session.expires_at > current_time
            }
            self.sessions = active_sessions
            
            sessions_data = [asdict(session) for session in self.sessions.values()]
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save sessions: {e}")
    
    def create_default_admin(self):
        """Create default admin user."""
        admin_user = self.create_user(
            username="admin",
            email="admin@example.com", 
            password="admin123",  # Should be changed in production
            role=Role.ADMIN
        )
        print(f"Created default admin user: {admin_user.username}")
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_user(self, username: str, email: str, password: str, 
                   role: Role = Role.USER, document_access: List[str] = None) -> User:
        """Create a new user."""
        
        # Check if username already exists
        if any(user.username == username for user in self.users.values()):
            raise ValueError(f"Username {username} already exists")
        
        # Check if email already exists
        if any(user.email == email for user in self.users.values()):
            raise ValueError(f"Email {email} already exists")
        
        user_id = hashlib.sha256(f"{username}:{email}:{time.time()}".encode()).hexdigest()[:16]
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            role=role,
            permissions=self.role_permissions[role].copy(),
            document_access=document_access or [],
            created_at=time.time(),
            last_login=None,
            is_active=True,
            metadata={}
        )
        
        self.users[user_id] = user
        self.save_users()
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username/password."""
        
        user = next((u for u in self.users.values() if u.username == username), None)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = time.time()
        self.save_users()
        
        return user
    
    def create_access_token(self, user: User, expires_delta: timedelta = None) -> str:
        """Create JWT access token."""
        
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        
        expire = datetime.now(timezone.utc) + expires_delta
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
    
    def create_session(self, user: User, ip_address: str = None, 
                      user_agent: str = None) -> Session:
        """Create a new user session."""
        
        session_id = secrets.token_urlsafe(32)
        expires_at = time.time() + (self.access_token_expire_minutes * 60)
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=time.time(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        self.save_sessions()
        
        return session
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_token(self, token: str) -> Optional[User]:
        """Get user from JWT token."""
        
        try:
            payload = self.verify_token(token)
            user_id = payload.get("sub")
            return self.get_user_by_id(user_id)
        except AuthenticationError:
            return None
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in user.permissions
    
    def can_access_document(self, user: User, document_id: str) -> bool:
        """Check if user can access a specific document."""
        
        # Admins can access all documents
        if user.role == Role.ADMIN:
            return True
        
        # Check document-specific access
        if document_id in user.document_access:
            return True
        
        # Check if user has general document read permission
        return Permission.READ_DOCUMENTS in user.permissions
    
    def grant_document_access(self, user_id: str, document_id: str) -> bool:
        """Grant user access to a document."""
        
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        if document_id not in user.document_access:
            user.document_access.append(document_id)
            self.save_users()
        
        return True
    
    def revoke_document_access(self, user_id: str, document_id: str) -> bool:
        """Revoke user access to a document."""
        
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        if document_id in user.document_access:
            user.document_access.remove(document_id)
            self.save_users()
        
        return True
    
    def update_user_role(self, user_id: str, new_role: Role) -> bool:
        """Update user role and permissions."""
        
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.role = new_role
        user.permissions = self.role_permissions[new_role].copy()
        self.save_users()
        
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account."""
        
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        self.save_users()
        
        # Invalidate all sessions for this user
        user_sessions = [sid for sid, session in self.sessions.items() if session.user_id == user_id]
        for session_id in user_sessions:
            del self.sessions[session_id]
        self.save_sessions()
        
        return True
    
    def get_all_users(self) -> List[User]:
        """Get all users (admin only)."""
        return list(self.users.values())


# Global user manager instance
_user_manager: UserManager | None = None


def get_user_manager() -> UserManager:
    """Get or create the global user manager."""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager


# FastAPI dependency for authentication
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_manager: UserManager = Depends(get_user_manager)
) -> User:
    """FastAPI dependency to get current authenticated user."""
    
    try:
        user = user_manager.get_user_by_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )
        
        return user
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        if permission not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        
        return current_user
    
    return permission_checker


def require_role(required_role: Role):
    """Decorator to require specific role or higher."""
    
    role_hierarchy = {
        Role.VIEWER: 1,
        Role.USER: 2,
        Role.MODERATOR: 3,
        Role.ADMIN: 4
    }
    
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled"
            )
        
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 999)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {required_role.value} or higher"
            )
        
        return current_user
    
    return role_checker


def require_document_access(document_id: str):
    """Decorator to require access to a specific document."""
    
    def access_checker(
        current_user: User = Depends(get_current_user),
        user_manager: UserManager = Depends(get_user_manager)
    ) -> User:
        
        if not user_manager.can_access_document(current_user, document_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
        
        return current_user
    
    return access_checker
