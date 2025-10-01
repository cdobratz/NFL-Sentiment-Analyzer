import pytest
from unittest.mock import MagicMock


class TestBasicFunctionality:
    """Test basic functionality without full app setup."""
    
    def test_sample_function(self):
        """Test a simple function to verify test setup works."""
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        result = add_numbers(2, 3)
        assert result == 5
    
    def test_mock_functionality(self):
        """Test mock functionality."""
        mock_service = MagicMock()
        mock_service.get_data.return_value = {"status": "success"}
        
        result = mock_service.get_data()
        assert result["status"] == "success"
        mock_service.get_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        async def async_add(a: int, b: int) -> int:
            return a + b
        
        result = await async_add(3, 4)
        assert result == 7