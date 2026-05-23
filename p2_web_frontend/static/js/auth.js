// JWT authentication helper
const Auth = {
    getToken() {
        return localStorage.getItem('access_token');
    },
    getRefreshToken() {
        return localStorage.getItem('refresh_token');
    },
    setTokens(access, refresh) {
        localStorage.setItem('access_token', access);
        if (refresh) localStorage.setItem('refresh_token', refresh);
    },
    clearTokens() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    },
    isLoggedIn() {
        return !!this.getToken();
    },
    async refreshAccessToken() {
        const refresh = this.getRefreshToken();
        if (!refresh) return false;
        try {
            const resp = await fetch('/api/v1/auth/refresh', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: refresh })
            });
            if (!resp.ok) return false;
            const data = await resp.json();
            if (data.code === 200 && data.data) {
                this.setTokens(data.data.access_token, data.data.refresh_token);
                return true;
            }
            return false;
        } catch {
            return false;
        }
    },
    async fetch(url, options = {}) {
        const token = this.getToken();
        const headers = { ...options.headers };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        if (!(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }
        let resp = await fetch(url, { ...options, headers });
        if (resp.status === 401 && this.getRefreshToken()) {
            const refreshed = await this.refreshAccessToken();
            if (refreshed) {
                headers['Authorization'] = `Bearer ${this.getToken()}`;
                resp = await fetch(url, { ...options, headers });
            } else {
                this.clearTokens();
                window.location.href = '/login';
            }
        }
        return resp;
    },
    redirectIfNotLoggedIn() {
        if (!this.isLoggedIn()) {
            window.location.href = '/login';
        }
    }
};
