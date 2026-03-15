
import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

# Load .env file from script directory
script_dir = Path(__file__).parent
env_file = script_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] | None = None,
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    if formats is None:
        formats = ['markdown', 'html']
    
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    app = FirecrawlApp(api_key=api_key)
    
    path = SCRAPE_DIR
    os.makedirs(path, exist_ok=True)
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    metadata_file = os.path.join(path, "scraped_metadata.json")
    
    # Load existing metadata if it exists
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    
    successful_scrapes = []
    
    for provider_name, url in websites.items():
        try:
            logger.info(f"Scraping {provider_name}: {url}")
            
            # Parse domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check if already scraped
            if provider_name in metadata:
                logger.info(f"Provider {provider_name} already scraped. Skipping...")
                successful_scrapes.append(provider_name)
                continue
            
            # Scrape the website
            scraped_result = app.scrape(url, formats=formats)
            
            # Convert Document to dict for easier access
            if hasattr(scraped_result, 'model_dump'):
                scraped_data = scraped_result.model_dump()
            else:
                scraped_data = scraped_result

            # Create metadata entry
            metadata_entry = {
                "provider_name": provider_name,
                "url": url,
                "domain": domain,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": True,
                "content_files": {},
                "title": scraped_data.get("metadata", {}).get("title", "") if isinstance(scraped_data, dict) else getattr(scraped_result, "title", ""),
                "description": scraped_data.get("metadata", {}).get("description", "") if isinstance(scraped_data, dict) else getattr(scraped_result, "description", "")
            }
            
            # Save content in each format
            for format_type in formats:
                content = None
                if isinstance(scraped_data, dict) and format_type in scraped_data:
                    content = scraped_data[format_type]
                elif hasattr(scraped_result, format_type):
                    content = getattr(scraped_result, format_type)
                
                if content:
                    filename = f"{provider_name}_{format_type}.txt"
                    filepath = os.path.join(path, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    metadata_entry["content_files"][format_type] = filename
                    logger.info(f"Saved {format_type} content for {provider_name}")
            
            metadata[provider_name] = metadata_entry
            successful_scrapes.append(provider_name)
            logger.info(f"Successfully scraped {provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to scrape {provider_name}: {str(e)}")
            metadata[provider_name] = {
                "provider_name": provider_name,
                "url": url,
                "scraped_at": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    # Save metadata to file
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
    except IOError as e:
        logger.error(f"Failed to save metadata: {str(e)}")
    
    total = len(websites)
    success_count = len(successful_scrapes)
    logger.info(f"Successfully scraped {success_count} out of {total} websites")
    return f"Successfully scraped {success_count} out of {total} websites: {successful_scrapes}"

@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR) if os.path.exists(SCRAPE_DIR) else 'directory does not exist'}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    logger.info(f"Checking metadata file: {metadata_file}")

    # Load metadata file
    if not os.path.exists(metadata_file):
        logger.warning(f"Metadata file not found: {metadata_file}")
        return json.dumps({"error": "No scraped data found", "identifier": identifier})
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        return json.dumps({"error": "Failed to load metadata", "details": str(e)})
    
    # Find matching entry by provider name, URL, or domain
    matching_entry = None
    
    # First try exact provider name match
    if identifier in metadata:
        matching_entry = metadata[identifier]
    else:
        # Try to match by URL or domain
        for provider_name, entry in metadata.items():
            if entry.get("success"):
                if identifier in entry.get("url", "") or identifier in entry.get("domain", ""):
                    matching_entry = entry
                    break
    
    if not matching_entry:
        logger.warning(f"No matching entry found for identifier: {identifier}")
        return json.dumps({
            "error": "No matching entry found",
            "identifier": identifier,
            "available_providers": list(metadata.keys())
        })
    
    # Load and include content if files exist
    if matching_entry.get("success"):
        content_files = matching_entry.get("content_files", {})
        for format_type, filename in content_files.items():
            filepath = os.path.join(SCRAPE_DIR, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Limit content length for response
                        matching_entry[f"{format_type}_content"] = content[:5000]  # First 5000 chars
                except IOError as e:
                    logger.error(f"Failed to read {filename}: {str(e)}")
                    matching_entry[f"{format_type}_content"] = f"Error reading file: {str(e)}"
    
    return json.dumps(matching_entry, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")