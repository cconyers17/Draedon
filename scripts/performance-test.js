// k6 performance test script for Text-to-CAD application
// Comprehensive load testing for all critical endpoints

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cadGenerationTime = new Trend('cad_generation_duration');
const uploadTime = new Trend('file_upload_duration');

// Test configuration
export const options = {
  stages: [
    // Ramp-up
    { duration: '2m', target: 10 }, // Ramp-up to 10 users over 2 minutes
    { duration: '5m', target: 10 }, // Stay at 10 users for 5 minutes
    { duration: '2m', target: 20 }, // Ramp-up to 20 users
    { duration: '5m', target: 20 }, // Stay at 20 users for 5 minutes
    { duration: '2m', target: 0 },  // Ramp-down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests must complete below 2s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],
    cad_generation_duration: ['p(95)<30000'], // CAD generation under 30s
    upload_time: ['p(95)<10000'],     // File uploads under 10s
  },
};

// Base URL from environment variable
const BASE_URL = __ENV.BASE_URL || 'https://text-to-cad-staging.onrender.com';

// Test data
const testDescriptions = [
  'Create a simple rectangular room 4 meters by 6 meters with 3 meter high ceiling',
  'Design a two-story house with living room, kitchen, and bedroom',
  'Build a commercial office space with multiple rooms and windows',
  'Create a modern apartment with open floor plan',
  'Design a warehouse with large open space and loading dock'
];

const testFiles = [
  // Small files for testing
  { name: 'small.txt', content: 'test content', size: 'small' },
  { name: 'medium.json', content: JSON.stringify({test: 'data'.repeat(1000)}), size: 'medium' },
];

// Setup function
export function setup() {
  console.log('Starting performance tests for Text-to-CAD application');
  console.log(`Base URL: ${BASE_URL}`);

  // Health check before starting
  const healthResponse = http.get(`${BASE_URL}/api/health`);
  if (healthResponse.status !== 200) {
    throw new Error('Application health check failed before testing');
  }

  console.log('Application is healthy, starting load tests...');
}

// Main test function
export default function() {
  const testUserId = `user_${__VU}_${__ITER}`;

  group('Health Checks', () => {
    // Frontend health check
    const frontendHealth = http.get(`${BASE_URL}/api/health`);
    check(frontendHealth, {
      'frontend health status is 200': (r) => r.status === 200,
      'frontend health response time < 1s': (r) => r.timings.duration < 1000,
    }) || errorRate.add(1);

    // Backend health check
    const backendHealth = http.get(`${BASE_URL}/api/health`);
    check(backendHealth, {
      'backend health status is 200': (r) => r.status === 200,
      'backend health contains status': (r) => r.json('status') !== undefined,
    }) || errorRate.add(1);
  });

  sleep(1);

  group('Text-to-CAD Generation', () => {
    const description = testDescriptions[Math.floor(Math.random() * testDescriptions.length)];

    const payload = {
      description: description,
      complexity_level: 'L1',
      material_preference: 'concrete',
      export_format: 'step'
    };

    const params = {
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': `k6-performance-test-${testUserId}`,
      },
      timeout: '60s',
    };

    const startTime = Date.now();
    const response = http.post(`${BASE_URL}/api/generate-cad`, JSON.stringify(payload), params);
    const generationDuration = Date.now() - startTime;

    cadGenerationTime.add(generationDuration);

    const success = check(response, {
      'CAD generation status is 200 or 202': (r) => [200, 202].includes(r.status),
      'CAD generation response time < 30s': (r) => r.timings.duration < 30000,
      'CAD generation returns job ID': (r) => {
        try {
          const body = r.json();
          return body.job_id !== undefined;
        } catch {
          return false;
        }
      },
    });

    if (!success) {
      errorRate.add(1);
      console.log(`CAD generation failed for user ${testUserId}: ${response.status} ${response.body}`);
    }

    // If we got a job ID, check the status
    if (response.status === 202) {
      try {
        const jobId = response.json('job_id');
        if (jobId) {
          sleep(2); // Wait before checking status

          const statusResponse = http.get(`${BASE_URL}/api/job-status/${jobId}`);
          check(statusResponse, {
            'job status check is successful': (r) => r.status === 200,
            'job status response time < 2s': (r) => r.timings.duration < 2000,
          });
        }
      } catch (e) {
        console.log(`Failed to check job status: ${e}`);
      }
    }
  });

  sleep(2);

  group('File Upload', () => {
    const testFile = testFiles[Math.floor(Math.random() * testFiles.length)];

    const formData = {
      file: http.file(testFile.content, testFile.name, 'text/plain'),
      description: 'Performance test upload',
    };

    const startTime = Date.now();
    const uploadResponse = http.post(`${BASE_URL}/api/upload`, formData, {
      timeout: '30s',
    });
    const uploadDuration = Date.now() - startTime;

    uploadTime.add(uploadDuration);

    check(uploadResponse, {
      'file upload status is 200 or 201': (r) => [200, 201].includes(r.status),
      'file upload response time < 10s': (r) => r.timings.duration < 10000,
    }) || errorRate.add(1);
  });

  sleep(1);

  group('Static Assets', () => {
    // Test static asset loading
    const staticRequests = [
      `${BASE_URL}/_next/static/css/app.css`,
      `${BASE_URL}/_next/static/js/main.js`,
      `${BASE_URL}/favicon.ico`,
    ];

    staticRequests.forEach(url => {
      const response = http.get(url, { timeout: '10s' });
      check(response, {
        [`${url} loads successfully`]: (r) => [200, 304].includes(r.status),
        [`${url} loads quickly`]: (r) => r.timings.duration < 2000,
      });
    });
  });

  sleep(1);

  group('API Endpoints', () => {
    // Test various API endpoints
    const endpoints = [
      { path: '/api/materials', name: 'materials list' },
      { path: '/api/building-codes', name: 'building codes' },
      { path: '/api/cad-templates', name: 'CAD templates' },
    ];

    endpoints.forEach(endpoint => {
      const response = http.get(`${BASE_URL}${endpoint.path}`, {
        timeout: '10s',
      });

      check(response, {
        [`${endpoint.name} endpoint responds successfully`]: (r) => r.status === 200,
        [`${endpoint.name} response time < 3s`]: (r) => r.timings.duration < 3000,
      });
    });
  });

  // Random sleep to simulate real user behavior
  sleep(Math.random() * 3 + 1);
}

// Teardown function
export function teardown(data) {
  console.log('Performance test completed');

  // Final health check
  const finalHealth = http.get(`${BASE_URL}/api/health`);
  if (finalHealth.status !== 200) {
    console.log('WARNING: Application health check failed after testing');
  } else {
    console.log('Application is still healthy after load testing');
  }
}

// Handle test stages differently
export function handleSummary(data) {
  return {
    'performance-results.json': JSON.stringify(data, null, 2),
    stdout: `
    =====================================================
    Text-to-CAD Performance Test Results
    =====================================================

    Total Requests: ${data.metrics.http_reqs.values.count}
    Failed Requests: ${data.metrics.http_req_failed.values.passes}
    Average Response Time: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms
    95th Percentile: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms

    CAD Generation:
    - Average Duration: ${data.metrics.cad_generation_duration ? data.metrics.cad_generation_duration.values.avg.toFixed(2) : 'N/A'}ms
    - 95th Percentile: ${data.metrics.cad_generation_duration ? data.metrics.cad_generation_duration.values['p(95)'].toFixed(2) : 'N/A'}ms

    File Uploads:
    - Average Duration: ${data.metrics.upload_time ? data.metrics.upload_time.values.avg.toFixed(2) : 'N/A'}ms
    - 95th Percentile: ${data.metrics.upload_time ? data.metrics.upload_time.values['p(95)'].toFixed(2) : 'N/A'}ms

    Error Rate: ${(data.metrics.errors ? data.metrics.errors.values.rate * 100 : 0).toFixed(2)}%

    =====================================================
    `,
  };
}