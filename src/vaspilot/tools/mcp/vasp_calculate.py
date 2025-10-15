import os
import time
import subprocess
import shutil
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import VaspInput, Vasprun, Kpoints, Poscar, Chgcar, Potcar, Outcar
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
import math
import pathlib
from typing import Optional, Dict, Any, Union
import numpy as np
from mcp.server.fastmcp import Context


def submit_slurm_job(calc_type: str, calculate_path: str, 
                     attachment_path: Optional[str] = None) -> Dict[str, Any]:
    """
    通用的SLURM任务提交方法
    
    参数:
        calculation_id: 计算ID
        calc_type: 计算类型 ("relaxation", "scf", "nscf")
        calculate_path: 计算路径
        attachment_path: 附件路径，包含SLURM脚本等文件
        
    返回:
        Dict包含slurm_id、calc_type、calculate_path、success、error、status等信息
    """
    try:
        # 如果指定了附件路径，复制附件文件到计算目录
        if attachment_path is not None and os.path.exists(attachment_path):
            # 复制附件目录下的所有文件到计算目录
            for file_name in os.listdir(attachment_path):
                src_file = os.path.join(attachment_path, file_name)
                dst_file = os.path.join(calculate_path, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        
        # 查找SLURM脚本文件
        slurm_script_path = None
        for script_name in ['submit.sh', 'run.sh', 'slurm.sh']:
            script_path = os.path.join(calculate_path, script_name)
            if os.path.exists(script_path):
                slurm_script_path = script_path
                break
        
        if slurm_script_path is None:
            return {
                "slurm_id": None,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "success": False,
                "error": "No SLURM script found (submit.sh, run.sh, or job.sh)",
                "status": "failed"
            }
        
        # 提交SLURM任务
        time.sleep(3)
        result = subprocess.run(['sbatch', slurm_script_path], 
                              capture_output=True, text=True, cwd=calculate_path)
        
        if result.returncode == 0:
            # 从sbatch输出中提取任务ID
            slurm_id = result.stdout.strip().split()[-1]
            
            return {
                "slurm_id": slurm_id,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "success": True,
                "error": None,
                "status": "submitted"
            }
        else:
            return {
                "slurm_id": None,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "success": False,
                "error": result.stderr,
                "status": "failed"
            }
            
    except Exception as e:
        return {
            "slurm_id": None,
            "calc_type": calc_type,
            "calculate_path": calculate_path,
            "success": False,
            "error": str(e),
            "status": "failed"
        }


def vasp_relaxation(calculation_id: str, work_dir: str, struct: Structure, 
                   kpoints: Kpoints, incar_dict: dict, attachment_path: Optional[str] = None, 
                   potcar_map: Optional[Dict] = None, submit_slurm: bool = True) -> Dict[str, Any]:
    """
    提交VASP结构优化计算任务
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        attachment_path: 附件路径，包含SLURM脚本等文件
        
    返回:
        Dict包含slurm_id、calc_type、calculate_path、success、error、status等信息
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # 创建VASP输入文件
    # 手动获取元素列表，确保顺序与POSCAR一致
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )
    
    # 准备结构优化目录
    rlx_dir = os.path.join(calc_dir, "rlx/")
    os.makedirs(rlx_dir, exist_ok=True)
    vasp_input.write_input(rlx_dir)
    
    # 提交SLURM任务
    if submit_slurm:
        return submit_slurm_job("relaxation", rlx_dir, attachment_path)
    else:
        return {
            "calc_type": "relaxation",
            "calculate_path": rlx_dir,
            "success": True,
            "error": None,
            "status": "prepared"
        }


def vasp_scf(calculation_id: str, work_dir: str, struct: Structure, 
            kpoints: Kpoints, incar_dict: dict, chgcar_path: Optional[str] = None, 
            wavecar_path: Optional[str] = None, attachment_path: Optional[str] = None, 
            potcar_map: Optional[Dict] = None, submit_slurm: bool = True) -> Dict[str, Any]:
    """
    提交VASP自洽场计算任务
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        chgcar_path: CHGCAR文件路径
        wavecar_path: WAVECAR文件路径
        attachment_path: 附件路径，包含SLURM脚本等文件
        
    返回:
        Dict包含slurm_id、calc_type、calculate_path、success、error、status等信息
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # 创建VASP输入文件
    # 手动获取元素列表，确保顺序与POSCAR一致
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )

    # 准备自洽场计算目录
    scf_dir = os.path.join(calc_dir, "scf/")
    os.makedirs(scf_dir, exist_ok=True)
    vasp_input.write_input(scf_dir)
    
    # 复制相关文件
    if chgcar_path is not None and os.path.exists(chgcar_path):
        shutil.copy2(chgcar_path, os.path.join(scf_dir, "CHGCAR"))
    if wavecar_path is not None and os.path.exists(wavecar_path):
        shutil.copy2(wavecar_path, os.path.join(scf_dir, "WAVECAR"))
    
    # 提交SLURM任务
    if submit_slurm:
        return submit_slurm_job("scf", scf_dir, attachment_path)
    else:
        return {
                "calc_type": "scf",
                "calculate_path": scf_dir,
                "success": True,
                "error": None,
                "status": "prepared"
            }


def vasp_nscf(calculation_id: str, work_dir: str, struct: Structure, 
             kpoints: Kpoints, incar_dict: dict, chgcar_path: str, 
             wavecar_path: Optional[str] = None, attachment_path: Optional[str] = None, 
             potcar_map: Optional[Dict] = None, submit_slurm: bool = True) -> Dict[str, Any]:
    """
    提交VASP非自洽场计算任务（能带计算）
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        chgcar_path: CHGCAR文件路径
        wavecar_path: WAVECAR文件路径
        attachment_path: 附件路径，包含SLURM脚本等文件
        potcar_map: POTCAR映射字典
        
    返回:
        Dict包含slurm_id、calc_type、calculate_path、success、error、status等信息
    """
    Name = calculation_id
    calc_dir = os.path.abspath(f'{work_dir}/{Name}')
    if potcar_map is None:
        potcar_map = {}
    # 创建VASP输入文件
    # 手动获取元素列表，确保顺序与POSCAR一致
    poscar = Poscar(struct)
    unique_species = []
    for species in poscar.structure.species:
        species: Element
        if unique_species:
            if species.symbol != unique_species[-1]:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        else:
            if species.symbol not in potcar_map:
                potcar_map[species.symbol] = species.symbol
            unique_species.append(species.symbol)
    potcar_symbols = []
    for symbol in unique_species:
        potcar_symbols.append(potcar_map[symbol])

    vasp_input = VaspInput(
        poscar=poscar,
        incar=incar_dict,
        kpoints=kpoints,
        potcar=Potcar(potcar_symbols)
    )
    
    # 准备能带计算目录
    band_dir = os.path.join(calc_dir, "band/")
    os.makedirs(band_dir, exist_ok=True)
    vasp_input.write_input(band_dir)
    
    # 复制相关文件
    if os.path.exists(chgcar_path):
        shutil.copy2(chgcar_path, os.path.join(band_dir, "CHGCAR"))
    if wavecar_path is not None and os.path.exists(wavecar_path):
        shutil.copy2(wavecar_path, os.path.join(band_dir, "WAVECAR"))
    
    # 提交SLURM任务
    if submit_slurm:
        return submit_slurm_job("nscf", band_dir, attachment_path)
    else:
        return {
            "calc_type": "nscf",
            "calculate_path": band_dir,
            "success": True,
            "error": None,
            "status": "prepared"
        }


def check_status(calc_dict: dict[str, dict[str, Any]]) -> Dict[str, Any]:
    """
    检查SLURM任务状态并返回计算结果
    
    参数:
        calc_dict: {calc_id: {"slurm_id": slurm_id, "calc_type": calc_type, "calculate_path": calculate_path, "status": status}}
        
    返回:
        Dict包含每个任务的状态和结果
    """
    
    for calc_id, job_info in calc_dict.items():
        slurm_id = job_info["slurm_id"]
        calc_type = job_info["calc_type"]
        calculate_path = job_info["calculate_path"]
        
        try:
            # 检查SLURM任务状态
            time.sleep(3)
            result = subprocess.run(['squeue', '-j', slurm_id, '--noheader'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                # 任务仍在运行
                job_status = "running"
                job_result = {}
            else:
                # 任务已完成，检查是否成功
                time.sleep(3)
                sacct_result = subprocess.run(['sacct', '-j', slurm_id, '--format=State', '--noheader'], 
                                            capture_output=True, text=True)
                
                if sacct_result.returncode == 0:
                    state = sacct_result.stdout.strip().split('\n')[0].strip()
                    if 'COMPLETED' in state:
                        job_status = "completed"
                        # 读取计算结果
                        job_result = _read_calculation_result(calc_type, calculate_path)
                    elif 'FAILED' in state:
                        err_str = """ -----------------------------------------------------------------------------
|                                                                             |
|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR      ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR       #       #       #      |
|     E        R   R    R   R    O     O  R   R                               |
|     E        R    R   R    R   O     O  R    R      ###     ###     ###     |
|     EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###     |"""
                        try:
                            if os.path.exists(os.path.join(calculate_path, "log")):
                                with open(os.path.join(calculate_path, "log"), "r") as f:
                                    log_content = f.read().split(err_str)[1]

                            else:
                                with open(os.path.join(calculate_path, "OUTCAR"), "r") as f:
                                    log_content = f.read().split(err_str)[1]
                        except:
                            log_content = f" SLURM job failed without any error message"
                        job_status = "failed"
                        job_result = {"error": f"{log_content}"}
                    elif state == "TIMEOUT":
                        job_status = "timeout"
                        job_result = {"error": f"SLURM job timed out"}
                    else:
                        job_status = state.lower()
                        job_result = {"error": f"SLURM job exited with state: {state}"}
                else:
                    job_status = "unknown"
                    job_result = {"error": "Cannot determine job status"}
            
            calc_dict[calc_id].update(job_result)
            calc_dict[calc_id]["status"] = job_status
            
        except Exception as e:
            calc_dict[calc_id] = {
                "slurm_id": slurm_id,
                "calc_type": calc_type,
                "calculate_path": calculate_path,
                "status": "unknown",
                "error": str(e)
            }
    
    return calc_dict


def _read_calculation_result(calc_type: str, calculate_path: str) -> Dict[str, Any]:
    """
    根据计算类型读取计算结果
    """
    try:
        if calc_type == "relaxation":
            # 读取结构优化结果
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            contcar = Poscar.from_file(os.path.join(calculate_path, "CONTCAR"))
            
            return {
                "structure": contcar.structure,
                "total_energy": vasprun.final_energy,
                "max_force": np.max(np.linalg.norm(vasprun.ionic_steps[-1]['forces'], axis=1)),
                "stress": vasprun.ionic_steps[-1]['stress'],
                "ionic_steps": len(vasprun.ionic_steps),
                "status": "completed"
            }
            
        elif calc_type == "scf":
            # 读取自洽场计算结果
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            
            return {
                "structure": vasprun.final_structure,
                "total_energy": vasprun.final_energy,
                "efermi": vasprun.efermi,
                "band_gap": vasprun.get_band_structure().get_band_gap(),
                "dos": vasprun.complete_dos,
                "eigen_values": vasprun.eigenvalues,
                "is_metal": vasprun.get_band_structure().is_metal(),
                "status": "completed"
            }
            
        elif calc_type == "nscf":
            # 读取能带计算结果
            vasprun = Vasprun(os.path.join(calculate_path, "vasprun.xml"))
            bs = vasprun.get_band_structure()
            
            return {
                "structure": vasprun.final_structure,
                "band_structure": bs,
                "efermi": vasprun.efermi,
                "dos": vasprun.complete_dos,
                "eigenvalues": vasprun.eigenvalues,
                "is_metal": bs.is_metal(),
                "band_gap": bs.get_band_gap(),
                "cbm": bs.get_cbm(),
                "vbm": bs.get_vbm(),
                "success": True
            }
        else:
            return {
                "success": False,
                "error": f"Unknown calculation type: {calc_type}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def cancel_slurm_job(slurm_id: str) -> Dict[str, Any]:
    """
    取消SLURM任务
    """
    try:
        subprocess.run(['scancel', slurm_id], capture_output=True, text=True)
        return {"success": True, "message": f"SLURM job {slurm_id} cancelled"}
    except Exception as e:
        return {"success": False, "error": str(e)}