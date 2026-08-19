"""
FYERS API OAuth2 Authentication Flow.

Usage:
    from fyers_auth import FyersAuth

    auth = FyersAuth(
        client_id="APP_ID-100",
        secret_key="your_secret",
        redirect_uri="http://127.0.0.1:8080"
    )

    # Step 1: Start local server and open browser
    auth.start_login()

    # Step 2: Wait for user to log in (blocks until callback received)
    auth_code = auth.wait_for_code(timeout=120)

    # Step 3: Exchange auth_code for access_token
    access_token = auth.get_access_token(auth_code)

    # Step 4: Save for future use
    auth.save_token(access_token)
"""

import os
import json
import time
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional


TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fyers_token.json")


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the FYERS OAuth callback."""

    auth_code = None
    server_ref = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        code = params.get("auth_code", [None])[0]
        if code:
            _CallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="background:#0a1a10;color:#00ff88;font-family:monospace;text-align:center;padding:60px">
                <h1>Login Successful!</h1>
                <p>Auth code received. You can close this tab.</p>
                <script>setTimeout(function(){window.close()},2000);</script>
                </body></html>
            """)
            # Shutdown server after receiving code
            if _CallbackHandler.server_ref:
                threading.Thread(target=_CallbackHandler.server_ref.shutdown, daemon=True).start()
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Error: No auth_code received</h1></body></html>")

    def log_message(self, format, *args):
        pass  # Suppress default logging


class FyersAuth:
    """
    FYERS OAuth2 authentication helper.

    Handles the full login flow:
    1. Opens FYERS login page in browser
    2. Captures auth_code via local HTTP server
    3. Exchanges auth_code for access_token
    4. Saves/loads tokens from disk
    """

    def __init__(self, client_id: str = "", secret_key: str = "",
                 redirect_uri: str = "http://127.0.0.1:8080"):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self._server = None
        self._auth_code = None

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.secret_key)

    def start_login(self, open_browser: bool = True) -> str:
        """
        Start the OAuth login flow.

        1. Starts a local HTTP server on the redirect_uri port
        2. Generates the FYERS auth URL
        3. Opens it in the default browser

        Returns:
            The auth URL that was opened
        """
        try:
            from fyers_apiv3 import fyersModel
        except ImportError:
            raise RuntimeError("fyers-apiv3 not installed. Run: pip install fyers-apiv3")

        # Parse port from redirect_uri
        parsed = urlparse(self.redirect_uri)
        port = parsed.port or 8080

        # Start local HTTP server
        _CallbackHandler.auth_code = None
        _CallbackHandler.server_ref = None

        self._server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        _CallbackHandler.server_ref = self._server

        # Run server in background thread
        server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        server_thread.start()

        # Generate auth URL
        session = fyersModel.SessionModel(
            client_id=self.client_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            state="scanner_auth"
        )

        auth_url = session.generate_authcode()

        if open_browser:
            webbrowser.open(auth_url)

        return auth_url

    def wait_for_code(self, timeout: int = 120) -> Optional[str]:
        """
        Wait for the auth_code to be received from the callback.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            auth_code string or None if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if _CallbackHandler.auth_code:
                self._auth_code = _CallbackHandler.auth_code
                return self._auth_code
            time.sleep(0.5)

        return None

    def get_access_token(self, auth_code: str) -> Optional[str]:
        """
        Exchange auth_code for access_token.

        Args:
            auth_code: The code received from FYERS callback

        Returns:
            access_token string or None on failure
        """
        try:
            from fyers_apiv3 import fyersModel

            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                state="scanner_auth"
            )

            session.set_token(auth_code)
            token_response = session.generate_token()

            if token_response and "access_token" in token_response:
                return token_response["access_token"]
            elif token_response and "code" in token_response:
                # Some versions return the token directly
                return token_response["code"]

            return None

        except Exception as e:
            print(f"Token exchange failed: {e}")
            return None

    def save_token(self, access_token: str):
        """Save access_token and config to disk."""
        data = {
            "client_id": self.client_id,
            "secret_key": self.secret_key,
            "redirect_uri": self.redirect_uri,
            "access_token": access_token,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_token() -> Optional[dict]:
        """Load saved token from disk."""
        if not os.path.exists(TOKEN_FILE):
            return None
        try:
            with open(TOKEN_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def clear_token():
        """Remove saved token."""
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)

    def cleanup(self):
        """Shutdown the local server if running."""
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass


def full_login_flow(client_id: str, secret_key: str,
                    redirect_uri: str = "http://127.0.0.1:8080",
                    timeout: int = 120) -> Optional[str]:
    """
    Run the complete login flow interactively.

    1. Opens browser to FYERS login
    2. Waits for callback
    3. Exchanges code for token
    4. Saves token to disk

    Returns:
        access_token or None
    """
    auth = FyersAuth(client_id, secret_key, redirect_uri)

    print("="*50)
    print("  FYERS Login Flow")
    print("="*50)
    print()

    # Start login
    auth_url = auth.start_login(open_browser=True)
    print(f"  Browser opened. Please log in to FYERS.")
    print(f"  If browser didn't open, visit:")
    print(f"  {auth_url[:80]}...")
    print()
    print(f"  Waiting for login (timeout: {timeout}s)...")

    # Wait for auth code
    auth_code = auth.wait_for_code(timeout=timeout)

    if not auth_code:
        print("  ✗ Login timed out or failed")
        auth.cleanup()
        return None

    print(f"  ✓ Auth code received!")

    # Exchange for access token
    print("  Exchanging auth code for access token...")
    access_token = auth.get_access_token(auth_code)

    if not access_token:
        print("  ✗ Token exchange failed")
        auth.cleanup()
        return None

    print(f"  ✓ Access token obtained! ({access_token[:20]}...)")

    # Save token
    auth.save_token(access_token)
    print(f"  ✓ Token saved to {TOKEN_FILE}")

    auth.cleanup()
    print()
    print("  Login complete! You can now use FYERS as data source.")
    return access_token


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python fyers_auth.py <client_id> <secret_key> [redirect_uri]")
        print("  Example: python fyers_auth.py APP_ID-100 my_secret http://127.0.0.1:8080")
        sys.exit(1)

    client_id = sys.argv[1]
    secret_key = sys.argv[2]
    redirect_uri = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080"

    token = full_login_flow(client_id, secret_key, redirect_uri)
    if token:
        print(f"\n  Access Token: {token}")
    else:
        print("\n  Login failed!")
        sys.exit(1)
