//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh.hpp"
#include "../fluid/fluid.hpp"
#include "outputs.hpp"

//======================================================================================
/*! \file outputs.cpp
 *  \brief implements functions for Athena++ outputs
 *
 * The number and types of outputs are all controlled by the number and values of
 * parameters specified in <outputN> blocks in the input file.  Each output block must
 * be labelled by a unique integer "N".  Following the convention of the parser
 * implemented in the ParameterInput class, a second output block with the same integer
 * "N" of an earlier block will silently overwrite the values read by the first block.
 * The numbering of the output blocks does not need to be consecutive, and blocks may
 * appear in any order in the input file.  Moreover, unlike the C version of Athena, the
 * total number of <outputN> blocks does not need to be specified -- in Athena++ a new
 * output type will be created for each and every <outputN> block in the input file.
 *
 * Required parameters that must be specified in an <outputN> block are:
 *   - variable     = cons,prim,D,d,E,e,m,v
 *   - file_type    = tab,vtk,hst
 *   - dt           = problem time between outputs
 *
 * Optional parameters that may be specified in an <outputN> block are:
 *   - data_format  = format string used in writing data (e.g. %12.5e)
 *   - next_time    = time of next output (useful for restarts)
 *   - id           = any string
 *   - file_number  = any integer with up to 4 digits
 *   - x[123]_slice = specifies data should be a slice at x[123] position
 *   - x[123]_sum   = set to 1 to sum data along specified direction
 *   
 * EXAMPLE of an <outputN> block for a VTK dump:
 *   <output3>
 *   file_type   = tab       # Tabular data dump
 *   variable    = prim      # variables to be output
 *   data_format = %12.5e    # Optional data format string
 *   dt          = 0.01      # time increment between outputs
 *   x2_slice    = 0.0       # slice in x2
 *   x3_slice    = 0.0       # slice in x3
 *
 * Each <outputN> block will result in a new node being created in a linked list of
 * OutputType stored in the Outputs class.  During a simulation, outputs are made when
 * the simulation time satisfies the criteria implemented in the MakeOutputs() function.
 *
 * To implement a new type of output X, write a new derived OutputType class, and
 * construct an object of this class in the Outputs::InitOutputTypes() function.
 *====================================================================================*/

//--------------------------------------------------------------------------------------
// OutputVariable constructor

OutputVariable::OutputVariable(AthenaArray<Real> *parray, OutputVariableHeader vhead)
{
  var_header = vhead;
  pdata = parray;
  pnext = NULL;
  pprev = NULL;
}

// destructor

OutputVariable::~OutputVariable()
{
  if (!pdata->IsShallowCopy()) pdata->DeleteAthenaArray();
}

//--------------------------------------------------------------------------------------
// OutputData constructor

OutputData::OutputData()
{
  pfirst_var = NULL;
  plast_var = NULL;
}

// destructor - iterates through linked list of OutputVariables and deletes nodes

OutputData::~OutputData()
{
  OutputVariable *pvar = pfirst_var;
  while(pvar != NULL) {
    OutputVariable *pvar_old = pvar;
    pvar = pvar->pnext;
    delete pvar_old;
  }
}

//--------------------------------------------------------------------------------------
// OutputType constructor

OutputType::OutputType(OutputParameters oparams)
{
  output_params = oparams;
  pnext = NULL; // Terminate linked list with NULL ptr
}

// destructor

OutputType::~OutputType()
{
}

//--------------------------------------------------------------------------------------
// Outputs constructor

Outputs::Outputs(Mesh *pm, ParameterInput *pin)
{
  pfirst_type_ = NULL;
  std::stringstream msg;
  InputBlock *pib = pin->pfirst_block;
  OutputType *pnew_type;
  OutputType *plast = pfirst_type_;

// loop over input block names.  Find those that start with "output", read parameters,
// and construct linked list of OutputTypes.

  while (pib != NULL) {
    if (pib->block_name.compare(0,6,"output") == 0) {
      OutputParameters op;  // define temporary OutputParameters struct

// extract integer number of output block.  Save name and number 

      std::string outn = pib->block_name.substr(6); // 6 because starts at 0!
      op.block_number = atoi(outn.c_str());
      op.block_name.assign(pib->block_name);

// set time of last output, time between outputs

      op.next_time = pin->GetOrAddReal(op.block_name,"next_time",0.0);
      op.dt = pin->GetReal(op.block_name,"dt");

// set file number, basename, id, and format

      op.file_number = pin->GetOrAddInteger(op.block_name,"file_number",0);
      op.file_basename = pin->GetString("job","problem_id");
      char define_id[10];
      sprintf(define_id,"out%d",op.block_number);  // default id="outN"
      op.file_id = pin->GetOrAddString(op.block_name,"id",define_id);
      op.file_type = pin->GetString(op.block_name,"file_type");

// read slicing options.  Check that slice is within mesh

      if (pin->DoesParameterExist(op.block_name,"x1_slice")) {
        Real x1 = pin->GetReal(op.block_name,"x1_slice");
        if (x1 >= pm->mesh_size.x1min && x1 < pm->mesh_size.x1max) {
          op.x1_slice = x1;
          op.islice = 1;
        } else {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Slice at x1=" << x1 << " in output block '" << op.block_name
              << "' is out of range of Mesh" << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      } else {
        op.islice = 0;
      }

      if (pin->DoesParameterExist(op.block_name,"x2_slice")) {
        Real x2 = pin->GetReal(op.block_name,"x2_slice");
        if (x2 >= pm->mesh_size.x2min && x2 < pm->mesh_size.x2max) {
          op.x2_slice = x2;
          op.jslice = 1;
        } else {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Slice at x2=" << x2 << " in output block '" << op.block_name
              << "' is out of range of Mesh" << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      } else {
        op.jslice = 0;
      }

      if (pin->DoesParameterExist(op.block_name,"x3_slice")) {
        Real x3 = pin->GetReal(op.block_name,"x3_slice");
        if (x3 >= pm->mesh_size.x3min && x3 < pm->mesh_size.x3max) {
          op.x3_slice = x3;
          op.kslice = 1;
        } else {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Slice at x3=" << x3 << " in output block '" << op.block_name
              << "' is out of range of Mesh" << std::endl;
          throw std::runtime_error(msg.str().c_str());
        }
      } else {
        op.kslice = 0;
      }

// read sum options.  Check for conflicts with slicing.

      if (pin->DoesParameterExist(op.block_name,"x1_sum")) {
        if (pin->DoesParameterExist(op.block_name,"x1_slice")) {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl 
              << "Cannot request both slice and sum along x1-direction"
              << " in output block '" << op.block_name << "'" << std::endl;
          throw std::runtime_error(msg.str().c_str());
        } else {
          op.isum = pin->GetInteger(op.block_name,"x1_sum");;
        }
      } else {
        op.isum = 0;
      }

      if (pin->DoesParameterExist(op.block_name,"x2_sum")) {
        if (pin->DoesParameterExist(op.block_name,"x2_slice")) {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Cannot request both slice and sum along x2-direction"
              << " in output block '" << op.block_name << "'" << std::endl;
          throw std::runtime_error(msg.str().c_str());
        } else {
          op.jsum = pin->GetInteger(op.block_name,"x2_sum");;
        }
      } else {
        op.jsum = 0;
      }

      if (pin->DoesParameterExist(op.block_name,"x3_sum")) {
        if (pin->DoesParameterExist(op.block_name,"x3_slice")) {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Cannot request both slice and sum along x3-direction"
              << " in output block '" << op.block_name << "'" << std::endl;
          throw std::runtime_error(msg.str().c_str());
        } else {
          op.ksum = pin->GetInteger(op.block_name,"x3_sum");;
        }
      } else {
        op.ksum = 0;
      }

// set output variable and optional data format string used in formatted writes

      if (op.file_type.compare("hst") != 0) {
        op.variable = pin->GetString(op.block_name,"variable");
      }
      op.data_format = pin->GetOrAddString(op.block_name,"data_format","%12e.5");
      op.data_format.insert(0," "); // prepend with blank to separate columns

// Construct new OutputType according to file format
// TODO: add any new output types here

      if (op.file_type.compare("tab") == 0) {
        pnew_type = new FormattedTableOutput(op);
      } else if (op.file_type.compare("hst") == 0) {
        pnew_type = new HistoryOutput(op);
      } else if (op.file_type.compare("vtk") == 0) {
        pnew_type = new VTKOutput(op);
      } else {
        msg << "### FATAL ERROR in function [Outputs::InitOutputTypes]"
            << std::endl << "Unrecognized file format = '" << op.file_type 
            << "' in output block '" << op.block_name << "'" << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }

// Add type as node in linked list 

      if (pfirst_type_ == NULL) {
        pfirst_type_ = pnew_type;
      } else {
        plast->pnext = pnew_type;
      }
      plast = pnew_type;
    }
    pib = pib->pnext;  // move to next input block name
  }
}

// destructor - iterates through linked list of OutputTypes and deletes nodes

Outputs::~Outputs()
{
  OutputType *ptype = pfirst_type_;
  while(ptype != NULL) {
    OutputType *ptype_old = ptype;
    ptype = ptype->pnext;
    delete ptype_old;
  }
}

//--------------------------------------------------------------------------------------
/*! \fn void OutputData::AppendNode()
 *  \brief
 */

void OutputData::AppendNode(AthenaArray<Real> *parray, OutputVariableHeader vhead)
{
  OutputVariable *pnew_var = new OutputVariable(parray, vhead);

  if (pfirst_var == NULL)
    pfirst_var = plast_var = pnew_var;
  else {
    pnew_var->pprev = plast_var;
    plast_var->pnext = pnew_var;
    plast_var = pnew_var;
  }
}

//--------------------------------------------------------------------------------------
/*! \fn void OutputData::ReplaceNode()
 *  \brief
 */

void OutputData::ReplaceNode(OutputVariable *pold, OutputVariable *pnew) 
{
  if (pold == pfirst_var) {
    pfirst_var = pnew;
    if (pold->pnext != NULL) {    // there is another node in the list 
      pnew->pnext = pold->pnext;
      pnew->pnext->pprev = pnew;
    } else {                      // there is only one node in the list
      plast_var = pnew;
    }
  } else if (pold == plast_var) {
    plast_var = pnew;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
  } else {
    pnew->pnext = pold->pnext;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
    pnew->pnext->pprev = pnew;
  }
  delete pold;
}

//--------------------------------------------------------------------------------------
/*! \fn void OutputType::LoadOutputData(OutputData *pod)
 *  \brief initializes output data in OutputData container
 */

void OutputType::LoadOutputData(OutputData *pod, MeshBlock *pmb)
{
  OutputVariableHeader var_header;
  Fluid *pf = pmb->pfluid;;
  std::stringstream str;

// Create OutputData header

  str << "# Athena++ data at time=" << pmb->pmy_domain->pmy_mesh->time
      << "  cycle=" << pmb->pmy_domain->pmy_mesh->ncycle
      << "  variables=" << output_params.variable << std::endl;
  pod->data_header.descriptor.append(str.str());
  pod->data_header.il = pmb->is;
  pod->data_header.iu = pmb->ie;
  pod->data_header.jl = pmb->js;
  pod->data_header.ju = pmb->je;
  pod->data_header.kl = pmb->ks;
  pod->data_header.ku = pmb->ke;
  pod->data_header.ndata = (pmb->ie - pmb->is + 1)*(pmb->je - pmb->js + 1)
                          *(pmb->ke - pmb->ks + 1);

// Create linked list of OutputVariables containing requested data

  int var_added = 0;
  if (output_params.variable.compare("D") == 0 || 
      output_params.variable.compare("cons") == 0) {
    var_header.type = "SCALARS";
    var_header.name = "dens";
    pod->AppendNode(pf->u.ShallowSlice(IDN,1),var_header); // (lab-frame) density
    var_added = 1;
  }

  if (output_params.variable.compare("d") == 0 || 
      output_params.variable.compare("prim") == 0) {
    var_header.type = "SCALARS";
    var_header.name = "rho";
    pod->AppendNode(pf->w.ShallowSlice(IDN,1),var_header); // (rest-frame) density
    var_added = 1;
  }

  if (NON_BAROTROPIC_EOS) {
    if (output_params.variable.compare("E") == 0 || 
        output_params.variable.compare("cons") == 0) {
      var_header.type = "SCALARS";
      var_header.name = "Etot";
      pod->AppendNode(pf->u.ShallowSlice(IEN,1),var_header); // total energy
      var_added = 1;
    }
  }

  if (NON_BAROTROPIC_EOS) {
    if (output_params.variable.compare("e") == 0 || 
        output_params.variable.compare("prim") == 0) {
      var_header.type = "SCALARS";
      var_header.name = "eint";
      pod->AppendNode(pf->w.ShallowSlice(IEN,1),var_header); // internal energy
      var_added = 1;
    }
  }

  if (output_params.variable.compare("m") == 0 || 
      output_params.variable.compare("cons") == 0) {
    var_header.type = "VECTORS";
    var_header.name = "mom";
    pod->AppendNode(pf->u.ShallowSlice(IM1,3),var_header); // momentum vector
    var_added = 1;
  }

  if (output_params.variable.compare("v") == 0 || 
      output_params.variable.compare("prim") == 0) {
    var_header.type = "VECTORS";
    var_header.name = "vel";
    pod->AppendNode(pf->w.ShallowSlice(IM1,3),var_header); // velocity vector
    var_added = 1;
  }

  if (output_params.variable.compare("ifov") == 0) {
    for (int n=0; n<(NIFOV); ++n) {
      var_header.type = "SCALARS";
      var_header.name = "ifov";
      pod->AppendNode(pf->ifov.ShallowSlice(n,1),var_header); // internal fluid outvars
    }
    var_added = 1;
  }

// throw an error if output variable name not recognized

  if (!var_added) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [OutputType::LoadOutputData]" << std::endl
        << "Output variable '" << output_params.variable << "' not implemented"
        << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  return;
}

//--------------------------------------------------------------------------------------
/*! \fn void OutputType::TransformOutputData()
 *  \brief 
 */

void OutputType::TransformOutputData(OutputData *pod, MeshBlock *pmb)
{
  if (output_params.kslice) {
    Slice(pod,pmb,3);
  }
  if (output_params.jslice) {
    Slice(pod,pmb,2);
  }
  if (output_params.islice) {
    Slice(pod,pmb,1);
  }
  if (output_params.ksum) {
    Sum(pod,pmb,3);
  }
  if (output_params.jsum) {
    Sum(pod,pmb,2);
  }
  if (output_params.isum) {
    Sum(pod,pmb,1);
  }
  return;
}

//--------------------------------------------------------------------------------------
/*! \fn void OutputType::Slice(OutputData* pod, int dim)
 *  \brief
 */

void OutputType::Slice(OutputData* pod, MeshBlock *pmb, int dim)
{
  int islice, jslice, kslice;

// Check that slice is in range of data in this block, if not return 0 in ndata

  if (dim == 1) {
    if (output_params.x1_slice >= pmb->block_size.x1min && 
        output_params.x1_slice < pmb->block_size.x1max) {
      for (int i=pmb->is+1; i<=pmb->ie+1; ++i) {
        if (pmb->x1f(i) > output_params.x1_slice) {
           islice = i-1;
          break;
        }
      }
    } else {
      pod->data_header.ndata = 0;
      return;
    }
  } else if (dim == 2) {
    if (output_params.x2_slice >= pmb->block_size.x2min &&
        output_params.x2_slice < pmb->block_size.x2max) {
      for (int j=pmb->js+1; j<=pmb->je+1; ++j) {
        if (pmb->x2f(j) > output_params.x2_slice) {
           jslice = j-1;
          break;
        }
      }
    } else {
      pod->data_header.ndata = 0;
      return;
    }
  } else {
    if (output_params.x3_slice >= pmb->block_size.x3min &&
        output_params.x3_slice < pmb->block_size.x3max) {
      for (int k=pmb->ks+1; k<=pmb->ke+1; ++k) {
        if (pmb->x3f(k) > output_params.x3_slice) {
           kslice = k-1;
          break;
        }
      }
    } else {
      pod->data_header.ndata = 0;
      return;
    }
  }

// For each node in OutputData linked list, slice arrays containing output data  

  OutputVariableHeader var_header;
  OutputVariable *pvar;
  pvar = pod->pfirst_var;
  AthenaArray<Real> *pslice;

  while (pvar != NULL) {
    var_header = pvar->var_header;
    int nx4 = pvar->pdata->GetDim4();
    int nx3 = pvar->pdata->GetDim3();
    int nx2 = pvar->pdata->GetDim2();
    int nx1 = pvar->pdata->GetDim1();
    pslice = new AthenaArray<Real>;

// Loop over variables and dimensions, extract slice

    if (dim == 3) {
      pslice->NewAthenaArray(nx4,1,nx2,nx1);
      for (int n=0; n<nx4; ++n){
      for (int j=(pod->data_header.jl); j<=(pod->data_header.ju); ++j){
        for (int i=(pod->data_header.il); i<=(pod->data_header.iu); ++i){
          (*pslice)(n,0,j,i) = (*pvar->pdata)(n,kslice,j,i);
        }
      }}
    } else if (dim == 2) {
      pslice->NewAthenaArray(nx4,nx3,1,nx1);
      for (int n=0; n<nx4; ++n){
      for (int k=(pod->data_header.kl); k<=(pod->data_header.ku); ++k){
        for (int i=(pod->data_header.il); i<=(pod->data_header.iu); ++i){
          (*pslice)(n,k,0,i) = (*pvar->pdata)(n,k,jslice,i);
        }
      }}
    } else {
      pslice->NewAthenaArray(nx4,nx3,nx2,1);
      for (int n=0; n<nx4; ++n){
      for (int k=(pod->data_header.kl); k<=(pod->data_header.ku); ++k){
        for (int j=(pod->data_header.jl); j<=(pod->data_header.ju); ++j){
          (*pslice)(n,k,j,0) = (*pvar->pdata)(n,k,j,islice);
        }
      }}
    }

    OutputVariable *pnew = new OutputVariable(pslice, var_header);
    pod->ReplaceNode(pvar,pnew);
    pvar = pvar->pnext;
  }
 
// modify OutputData header

  std::stringstream str;
  if (dim == 3) {
    str << "# Slice at x3=" << pmb->x3v(output_params.kslice)
        << "  (k-ks)=" << (output_params.kslice - pmb->ks) << std::endl;
    pod->data_header.kl = 0;
    pod->data_header.ku = 0;
  } else if (dim == 2) {
    str << "# Slice at x2=" << pmb->x2v(output_params.jslice)
        << "  (j-js)=" << (output_params.jslice - pmb->js) << std::endl;
    pod->data_header.jl = 0;
    pod->data_header.ju = 0;
  } else {
    str << "# Slice at x1=" << pmb->x1v(output_params.islice)
        << "  (i-is)=" << (output_params.islice - pmb->is) << std::endl;
    pod->data_header.il = 0;
    pod->data_header.iu = 0;
  }
  pod->data_header.transforms.append(str.str());

  return;
}

//--------------------------------------------------------------------------------------
/*! \fn void OutputType::Sum(OutputData* pod, int dim)
 *  \brief
 */

void OutputType::Sum(OutputData* pod, MeshBlock* pmb, int dim)
{
  OutputVariableHeader var_header;
  AthenaArray<Real> *psum;
  std::stringstream str;

// For each node in OutputData linked list, sum arrays containing output data  

  OutputVariable *pvar;
  pvar = pod->pfirst_var;

  while (pvar != NULL) {
    var_header = pvar->var_header;
    int nx4 = pvar->pdata->GetDim4();
    int nx3 = pvar->pdata->GetDim3();
    int nx2 = pvar->pdata->GetDim2();
    int nx1 = pvar->pdata->GetDim1();
    psum = new AthenaArray<Real>;

// Loop over variables and dimensions, sum over specified dimension

    if (dim == 3) {
      psum->NewAthenaArray(nx4,1,nx2,nx1);
      for (int n=0; n<nx4; ++n){
      for (int k=(pod->data_header.kl); k<=(pod->data_header.ku); ++k){
      for (int j=(pod->data_header.jl); j<=(pod->data_header.ju); ++j){
        for (int i=(pod->data_header.il); i<=(pod->data_header.iu); ++i){
          (*psum)(n,0,j,i) += (*pvar->pdata)(n,k,j,i);
        }
      }}}
    } else if (dim == 2) {
      psum->NewAthenaArray(nx4,nx3,1,nx1);
      for (int n=0; n<nx4; ++n){
      for (int k=(pod->data_header.kl); k<=(pod->data_header.ku); ++k){
      for (int j=(pod->data_header.jl); j<=(pod->data_header.ju); ++j){
        for (int i=(pod->data_header.il); i<=(pod->data_header.iu); ++i){
          (*psum)(n,k,0,i) += (*pvar->pdata)(n,k,j,i);
        }
      }}}
    } else {
      psum->NewAthenaArray(nx4,nx3,nx2,1);
      for (int n=0; n<nx4; ++n){
      for (int k=(pod->data_header.kl); k<=(pod->data_header.ku); ++k){
      for (int j=(pod->data_header.jl); j<=(pod->data_header.ju); ++j){
        for (int i=(pod->data_header.il); i<=(pod->data_header.iu); ++i){
          (*psum)(n,k,j,0) += (*pvar->pdata)(n,k,j,i);
        }
      }}}
    }

    OutputVariable *pnew = new OutputVariable(psum, var_header);
    pod->ReplaceNode(pvar,pnew);
    pvar = pvar->pnext;
  }
 
// modify OutputData header

  if (dim == 3) {
    str << "# Sum over x3" << std::endl;
    pod->data_header.kl = 0;
    pod->data_header.ku = 0;
  } else if (dim == 2) {
    str << "# Sum over x2" << std::endl;
    pod->data_header.jl = 0;
    pod->data_header.ju = 0;
  } else {
    str << "# Sum over x1" << std::endl;
    pod->data_header.il = 0;
    pod->data_header.iu = 0;
  }
  pod->data_header.transforms.append(str.str());

  return;
}

//--------------------------------------------------------------------------------------
/*! \fn void Outputs::MakeOutputs()
 *  \brief scans through linked list of OutputTypes and makes any outputs needed.
 */

void Outputs::MakeOutputs(Mesh *pm)
{
  OutputType* ptype = pfirst_type_;

// Eventually this will be a loop over all domains

  MeshBlock *pmb = pm->pdomain->pblock;
  if (pmb != NULL)  {

    while (ptype != NULL) {
      if ((pm->time == pm->start_time) ||
          (pm->time >= ptype->output_params.next_time) ||
          (pm->time >= pm->tlim)) {

// Create new OutputData container, load and transform data, then write to file

        OutputData* pod = new OutputData;
        ptype->LoadOutputData(pod,pmb);
        ptype->TransformOutputData(pod,pmb);
        ptype->WriteOutputFile(pod,pmb);
        delete pod;

      }
      ptype = ptype->pnext; // move to next OutputType in list
    }
  }
}