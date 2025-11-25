import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API methods
export const apiClient = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Dashboard
  getDashboard: () => api.get('/dashboard'),

  // Projects
  createProject: (specSheet) => api.post('/projects', specSheet),
  getProjects: (status = null) => 
    api.get('/projects', { params: status ? { status } : {} }),
  getProject: (projectId) => api.get(`/projects/${projectId}`),
  getProjectStatus: (projectId) => api.get(`/projects/${projectId}/status`),
  getProjectReport: (projectId) => api.get(`/projects/${projectId}/report`),
  deleteProject: (projectId) => api.delete(`/projects/${projectId}`),

  // Spec sheet templates
  getSpecSheetTemplates: () => api.get('/spec-sheets'),

  // Documentation
  getDocumentation: () => api.get('/documentation'),
};

export default api;

