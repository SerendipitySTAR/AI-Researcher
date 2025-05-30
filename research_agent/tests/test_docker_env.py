# research_agent/tests/test_docker_env.py
import unittest
from unittest.mock import MagicMock, patch
# Assuming DockerEnv and DockerConfig are in research_agent.inno.environment.docker_env
from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig 

class TestDockerEnvGpuInfo(unittest.TestCase):

    def setUp(self):
        # Basic config for DockerEnv, can be dummy for these tests
        self.config = DockerConfig(
            container_name="test_container",
            workplace_name="test_workplace",
            communication_port=12345,
            conda_path="/root/miniconda3"
        )
        self.docker_env = DockerEnv(self.config)

    @patch.object(DockerEnv, 'run_command')
    def test_get_gpu_memory_info_success_single_gpu(self, mock_run_command):
        # Mock the output of nvidia-smi
        mock_run_command.return_value = {
            'status': 0,
            'result': "48000,24000\n" # Total, Free in MiB
        }
        
        gpu_info = self.docker_env.get_gpu_memory_info()
        
        self.assertEqual(len(gpu_info), 1)
        self.assertEqual(gpu_info[0]['id'], 0)
        self.assertEqual(gpu_info[0]['total_memory_mib'], 48000)
        self.assertEqual(gpu_info[0]['free_memory_mib'], 24000)
        mock_run_command.assert_called_once_with(
            "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits"
        )

    @patch.object(DockerEnv, 'run_command')
    def test_get_gpu_memory_info_success_multiple_gpus(self, mock_run_command):
        mock_run_command.return_value = {
            'status': 0,
            'result': "48000,24000\n16000,8000\n"
        }
        gpu_info = self.docker_env.get_gpu_memory_info()
        self.assertEqual(len(gpu_info), 2)
        self.assertEqual(gpu_info[0]['total_memory_mib'], 48000)
        self.assertEqual(gpu_info[0]['free_memory_mib'], 24000)
        self.assertEqual(gpu_info[1]['id'], 1)
        self.assertEqual(gpu_info[1]['total_memory_mib'], 16000)
        self.assertEqual(gpu_info[1]['free_memory_mib'], 8000)

    @patch.object(DockerEnv, 'run_command')
    def test_get_gpu_memory_info_nvidia_smi_failure(self, mock_run_command):
        mock_run_command.return_value = {'status': 1, 'result': "Error"}
        gpu_info = self.docker_env.get_gpu_memory_info()
        self.assertEqual(len(gpu_info), 0)

    @patch.object(DockerEnv, 'run_command')
    def test_get_gpu_memory_info_empty_output(self, mock_run_command):
        mock_run_command.return_value = {'status': 0, 'result': ""}
        gpu_info = self.docker_env.get_gpu_memory_info()
        self.assertEqual(len(gpu_info), 0)

    @patch.object(DockerEnv, 'run_command')
    def test_get_gpu_memory_info_malformed_line(self, mock_run_command):
        mock_run_command.return_value = {'status': 0, 'result': "48000,24000\nmalformed_line\n16000,8000"}
        gpu_info = self.docker_env.get_gpu_memory_info()
        self.assertEqual(len(gpu_info), 2) # Should skip the malformed line
        self.assertEqual(gpu_info[0]['total_memory_mib'], 48000)
        self.assertEqual(gpu_info[1]['total_memory_mib'], 16000)
        
    @patch.object(DockerEnv, 'run_command')
    def test_get_gpu_memory_info_exception_in_run_command(self, mock_run_command):
        mock_run_command.side_effect = Exception("Docker connection failed")
        gpu_info = self.docker_env.get_gpu_memory_info()
        self.assertEqual(len(gpu_info), 0)


if __name__ == '__main__':
    unittest.main()
