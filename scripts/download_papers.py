"""
Download All Referenced Research Papers
Automatically downloads all papers mentioned in the research documents
"""

import os
import requests
import time
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# All papers with their download URLs
PAPERS = {
    # Foundation Models & Primary Methods
    "Sapiens_2B_ECCV2024": {
        "url": "https://arxiv.org/pdf/2408.12569.pdf",
        "title": "Sapiens: Foundation for Human Vision Models",
        "venue": "ECCV 2024 Oral",
        "priority": 1
    },
    "ViTPose_NeurIPS2022": {
        "url": "https://arxiv.org/pdf/2204.12004.pdf",
        "title": "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation",
        "venue": "NeurIPS 2022",
        "priority": 1
    },
    "DWPose_ICCV2023": {
        "url": "https://arxiv.org/pdf/2307.11573.pdf",
        "title": "Effective Whole-body Pose Estimation with Two-stages Distillation",
        "venue": "ICCV 2023",
        "priority": 1
    },
    
    # SSL & Advanced Methods
    "SSL_MultiPath_ICLR2025": {
        "url": "https://openreview.net/pdf?id=5zGuFj0y9V",
        "title": "Boosting Semi-Supervised 2D Human Pose Estimation by Revisiting Data Augmentation and Consistency Training",
        "venue": "ICLR 2025",
        "priority": 1
    },
    "SDPose_Diffusion": {
        "url": "https://arxiv.org/pdf/2509.24980.pdf",
        "title": "SDPose: Tokenized Pose Estimation via Circulation-Guide Self-Distillation",
        "venue": "Pre-print",
        "priority": 2
    },
    "UniPose_Multimodal": {
        "url": "https://arxiv.org/pdf/2411.16781.pdf",
        "title": "UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing",
        "venue": "Pre-print",
        "priority": 2
    },
    
    # Classical & Baseline Methods
    "DeepPose_CVPR2014": {
        "url": "https://arxiv.org/pdf/1312.4659.pdf",
        "title": "DeepPose: Human Pose Estimation via Deep Neural Networks",
        "venue": "CVPR 2014",
        "priority": 3
    },
    "OpenPose_CVPR2017": {
        "url": "https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf",
        "title": "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields",
        "venue": "CVPR 2017",
        "priority": 2
    },
    "HRNet_CVPR2019": {
        "url": "https://arxiv.org/pdf/1902.09212.pdf",
        "title": "Deep High-Resolution Representation Learning for Visual Recognition",
        "venue": "CVPR 2019",
        "priority": 2
    },
    
    # Datasets & Surveys
    "COCO_Dataset_ECCV2014": {
        "url": "https://arxiv.org/pdf/1405.0312.pdf",
        "title": "Microsoft COCO: Common Objects in Context",
        "venue": "ECCV 2014",
        "priority": 2
    },
    "Pose_Survey_2022": {
        "url": "https://arxiv.org/pdf/2204.07370.pdf",
        "title": "Deep Learning-Based Human Pose Estimation: A Survey",
        "venue": "2022 Survey",
        "priority": 3
    },
    
    # Additional Important Papers
    "SimpleBaseline_ECCV2018": {
        "url": "https://arxiv.org/pdf/1804.06208.pdf",
        "title": "Simple Baselines for Human Pose Estimation and Tracking",
        "venue": "ECCV 2018",
        "priority": 3
    },
    "HourglassNetworks_ECCV2016": {
        "url": "https://arxiv.org/pdf/1603.06937.pdf",
        "title": "Stacked Hourglass Networks for Human Pose Estimation",
        "venue": "ECCV 2016",
        "priority": 3
    },
    "CPM_CVPR2016": {
        "url": "https://arxiv.org/pdf/1602.00134.pdf",
        "title": "Convolutional Pose Machines",
        "venue": "CVPR 2016",
        "priority": 3
    },
    
    # Knowledge Distillation & Training
    "KnowledgeDistillation_Hinton": {
        "url": "https://arxiv.org/pdf/1503.02531.pdf",
        "title": "Distilling the Knowledge in a Neural Network",
        "venue": "NIPS 2014 Workshop",
        "priority": 3
    },
    
    # Vision Transformers (Foundation)
    "ViT_ICLR2021": {
        "url": "https://arxiv.org/pdf/2010.11929.pdf",
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "venue": "ICLR 2021",
        "priority": 2
    },
    "MAE_CVPR2022": {
        "url": "https://arxiv.org/pdf/2111.06377.pdf",
        "title": "Masked Autoencoders Are Scalable Vision Learners",
        "venue": "CVPR 2022",
        "priority": 2
    },
}


def download_paper(paper_id, paper_info, output_dir):
    """
    Download a single paper with retry logic
    
    Args:
        paper_id: Unique identifier for the paper
        paper_info: Dict with url, title, venue, priority
        output_dir: Directory to save the paper
    """
    url = paper_info['url']
    title = paper_info['title']
    venue = paper_info['venue']
    
    # Create filename
    filename = f"{paper_id}.pdf"
    filepath = output_dir / filename
    
    # Skip if already downloaded
    if filepath.exists():
        logger.info(f"✓ Already downloaded: {paper_id}")
        return True
    
    # Download with progress bar
    try:
        logger.info(f"📥 Downloading: {title}")
        logger.info(f"   Venue: {venue}")
        logger.info(f"   URL: {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=paper_id) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✓ Successfully downloaded: {paper_id}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to download {paper_id}: {e}")
        
        # Delete partial file
        if filepath.exists():
            filepath.unlink()
        
        return False
    
    except Exception as e:
        logger.error(f"❌ Unexpected error downloading {paper_id}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_all_papers(output_dir="papers", priority_filter=None):
    """
    Download all papers, optionally filtered by priority
    
    Args:
        output_dir: Directory to save papers
        priority_filter: Only download papers with this priority or lower (1=highest)
                        None = download all
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("RESEARCH PAPER DOWNLOADER")
    logger.info("="*60)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Total papers: {len(PAPERS)}")
    
    if priority_filter:
        papers_to_download = {k: v for k, v in PAPERS.items() 
                             if v['priority'] <= priority_filter}
        logger.info(f"Priority filter: ≤{priority_filter} ({len(papers_to_download)} papers)")
    else:
        papers_to_download = PAPERS
        logger.info("Priority filter: None (downloading all)")
    
    logger.info("="*60 + "\n")
    
    # Sort by priority
    sorted_papers = sorted(papers_to_download.items(), 
                          key=lambda x: x[1]['priority'])
    
    # Download papers
    success_count = 0
    failed_papers = []
    
    for i, (paper_id, paper_info) in enumerate(sorted_papers, 1):
        logger.info(f"\n[{i}/{len(sorted_papers)}] Processing: {paper_id}")
        logger.info(f"Priority: {paper_info['priority']}")
        
        success = download_paper(paper_id, paper_info, output_path)
        
        if success:
            success_count += 1
        else:
            failed_papers.append((paper_id, paper_info))
        
        # Rate limiting (be nice to servers)
        if i < len(sorted_papers):
            time.sleep(2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"✓ Successfully downloaded: {success_count}/{len(sorted_papers)}")
    logger.info(f"❌ Failed: {len(failed_papers)}")
    
    if failed_papers:
        logger.info("\nFailed downloads:")
        for paper_id, paper_info in failed_papers:
            logger.info(f"  - {paper_id}: {paper_info['url']}")
    
    logger.info(f"\nPapers saved to: {output_path.absolute()}")
    logger.info("="*60 + "\n")
    
    return success_count, failed_papers


def create_bibliography(output_file="BIBLIOGRAPHY.md"):
    """Create a formatted bibliography file"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Complete Bibliography - Pose LLM Identifier\n\n")
        f.write("## Primary Methods (Priority 1)\n\n")
        
        for priority in [1, 2, 3]:
            if priority > 1:
                f.write(f"\n## {'Supporting Methods' if priority == 2 else 'Background & Surveys'} (Priority {priority})\n\n")
            
            priority_papers = {k: v for k, v in PAPERS.items() if v['priority'] == priority}
            
            for paper_id, info in sorted(priority_papers.items()):
                f.write(f"### {paper_id}\n")
                f.write(f"- **Title**: {info['title']}\n")
                f.write(f"- **Venue**: {info['venue']}\n")
                f.write(f"- **URL**: {info['url']}\n")
                f.write(f"- **Local File**: `papers/{paper_id}.pdf`\n\n")
        
        f.write("\n---\n")
        f.write(f"Total papers: {len(PAPERS)}\n")
    
    logger.info(f"✓ Bibliography created: {output_file}")


def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download research papers")
    parser.add_argument('--output', '-o', default='papers', 
                       help='Output directory (default: papers)')
    parser.add_argument('--priority', '-p', type=int, default=None,
                       help='Only download priority ≤ N (1=essential, 2=important, 3=background)')
    parser.add_argument('--essential-only', action='store_true',
                       help='Only download essential papers (priority 1)')
    parser.add_argument('--bibliography', '-b', action='store_true',
                       help='Create bibliography file')
    
    args = parser.parse_args()
    
    # Set priority filter
    if args.essential_only:
        priority_filter = 1
    else:
        priority_filter = args.priority
    
    # Create bibliography
    if args.bibliography:
        create_bibliography()
    
    # Download papers
    success_count, failed_papers = download_all_papers(
        output_dir=args.output,
        priority_filter=priority_filter
    )
    
    # Print quick reference
    if success_count > 0:
        logger.info("\n📚 QUICK REFERENCE:")
        logger.info("   Essential papers (Priority 1): Sapiens-2B, ViTPose, DWPose, SSL")
        logger.info("   Important papers (Priority 2): HRNet, OpenPose, ViT, MAE, COCO")
        logger.info("   Background papers (Priority 3): DeepPose, Hourglass, CPM, Surveys")
        logger.info("\n💡 TIP: Start reading with Priority 1 papers!")
    
    # Retry failed downloads
    if failed_papers and len(failed_papers) < 5:
        logger.info("\n🔄 Retrying failed downloads...")
        time.sleep(5)
        
        output_path = Path(args.output)
        retry_success = 0
        
        for paper_id, paper_info in failed_papers:
            if download_paper(paper_id, paper_info, output_path):
                retry_success += 1
            time.sleep(2)
        
        if retry_success > 0:
            logger.info(f"✓ Successfully downloaded {retry_success} papers on retry")


if __name__ == "__main__":
    main()
